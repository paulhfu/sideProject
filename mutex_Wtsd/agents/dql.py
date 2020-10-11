from data.disjoint_discs import MultiDiscSpGraphDset
from data.disjoint_discs_balanced_graph import MultiDiscSpGraphDsetBalanced
import torch
from torch.optim import Adam
import time
from torch import nn
from models.sp_embed_unet import SpVecsUnet
from torch.nn import functional as F
from torch.utils.data import DataLoader
from agents.replayMemory import TransitionData
from collections import namedtuple
from losses import ContrastiveLoss
from environments.sp_grph_gcn_1 import SpGcnEnv
from models.GCNNs.q_value_net import GcnEdgeAngle1dQ
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tensorboardX import SummaryWriter
import os
from optimizers.adam import CstmAdam
from agents.exploration_functions import ExponentialAverage, FollowLeadAvg, FollowLeadMin, RunningAverage, \
    ActionPathTreeNodes, ExpSawtoothEpsDecay, NaiveDecay, GaussianDecay, Constant
import numpy as np
from utils.general import adjust_learning_rate
import matplotlib.pyplot as plt
import yaml


class AgentDqlTrainer(object):

    def __init__(self, args, global_count, global_writer_loss_count, global_writer_quality_count,
                 global_win_event_count, action_stats_count, save_dir):
        super(AgentDqlTrainer, self).__init__()

        self.args = args
        self.global_count = global_count
        self.global_writer_loss_count = global_writer_loss_count
        self.global_writer_quality_count = global_writer_quality_count
        self.global_win_event_count = global_win_event_count
        self.action_stats_count = action_stats_count
        self.writer_idx_warmup_loss = 0
        # self.eps = self.args.init_epsilon
        self.save_dir = save_dir
        if args.stop_qual_rule == 'naive':
            self.stop_qual_rule = NaiveDecay(initial_eps=args.init_stop_qual, episode_shrinkage=1,
                                             change_after_n_episodes=5)
        elif args.stop_qual_rule == 'gaussian':
            self.stop_qual_rule = GaussianDecay(args.stop_qual_final, args.stop_qual_scaling, args.stop_qual_offset,
                                                args.T_max)
        elif args.stop_qual_rule == 'running_average':
            self.stop_qual_rule = RunningAverage(args.stop_qual_ra_bw, args.stop_qual_scaling + args.stop_qual_offset,
                                                 args.stop_qual_ra_off)
        else:
            self.stop_qual_rule = NaiveDecay(args.init_stop_qual)

        if self.args.eps_rule == "treesearch":
            self.eps_rule = ActionPathTreeNodes()
        elif self.args.eps_rule == "sawtooth":
            self.eps_rule = ExpSawtoothEpsDecay()
        elif self.args.eps_rule == 'gaussian':
            self.eps_rule = GaussianDecay(args.eps_final, args.eps_scaling, args.eps_offset, args.T_max)
        elif self.args.eps_rule == "self_reg_min":
            self.args.T_max = np.inf
            self.eps_rule = FollowLeadMin((args.stop_qual_scaling + args.stop_qual_offset), 1)
        elif self.args.eps_rule == "self_reg_avg":
            self.args.T_max = np.inf
            self.eps_rule = FollowLeadAvg(1.5 * (args.stop_qual_scaling + args.stop_qual_offset), 2, 1)
        elif self.args.eps_rule == "self_reg_exp_avg":
            self.args.T_max = np.inf
            self.eps_rule = ExponentialAverage(1 * (args.stop_qual_scaling + args.stop_qual_offset), 0.9, 1)
        else:
            self.eps_rule = NaiveDecay(self.eps, 0.00005, 1000, 1)

    def setup(self, rank, world_size):
        # BLAS setup
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

        # os.environ["CUDA_VISIBLE_DEVICES"] = "6"
        assert torch.cuda.device_count() == 1
        torch.set_default_tensor_type('torch.FloatTensor')
        # Detect if we have a GPU available
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = self.args.master_port
        # os.environ['GLOO_SOCKET_IFNAME'] = 'eno1'

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # Explicitly setting seed to make sure that models created in two processes
        # start from same random weights and biases.
        torch.manual_seed(self.args.seed)

    def cleanup(self):
        dist.destroy_process_group()

    def update_env_data(self, env, dloader, device):
        edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes, affinities, gt = \
            next(iter(dloader))
        angles = None
        edges, edge_feat, gt_edge_weights, node_labeling, raw, nodes, affinities, gt = \
            edges.squeeze().to(device), edge_feat.squeeze()[:, 0:self.args.n_edge_features].to(device), \
            gt_edge_weights.squeeze().to(device), node_labeling.squeeze().to(device), raw.squeeze().to(
                device), nodes.squeeze().to(device), \
            affinities.squeeze().numpy(), gt.squeeze()
        env.update_data(edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes,
                        angles, affinities, gt)

    # Updates networks
    def _update_networks(self, loss, optimizer, shared_model, writer=None):
        # Zero shared and local grads
        optimizer.zero_grad()
        """
        Calculate gradients for gradient descent on loss functions
        Note that math comments follow the paper, which is formulated for gradient ascent
        """
        loss.backward()
        # Gradient L2 normalisation
        nn.utils.clip_grad_norm_(shared_model.parameters(), self.args.max_gradient_norm)
        optimizer.step()
        with open(os.path.join(self.save_dir, 'runtime_cfg.yaml')) as info:
            args_dict = yaml.full_load(info)
            if args_dict is not None:
                if 'lr' in args_dict:
                    if self.args.lr != args_dict['lr']:
                        print("lr changed from ", self.args.lr, " to ", args_dict['lr'], " at loss step ",
                              self.global_writer_loss_count.value())
                        self.args.lr = args_dict['lr']
        self.args.lr = args_dict['lr']
        new_lr = self.args.lr
        if self.args.min_lr != 0 and self.eps <= 0.6:
            # Linearly decay learning rate
            # new_lr = self.args.lr - ((self.args.lr - self.args.min_lr) * (1 - (self.eps * 2))) # (1 - max((self.args.T_max - self.global_count.value()) / self.args.T_max, 1e-32)))
            new_lr = self.args.lr * 10 ** (-(0.6 - self.eps))

        adjust_learning_rate(optimizer, new_lr)
        if writer is not None:
            writer.add_scalar("loss/learning_rate", new_lr, self.global_writer_loss_count.value())

    def get_action(self, q, policy, waff_dis, device, writer=None):

        action_probs = torch.softmax(q, -1)
        if policy == 'off_sampled':
            action_loc = action_probs.float().mean()
            behav_probs = action_probs.detach() + self.eps * (1 / self.args.n_actions - action_probs.detach())
            actions = torch.multinomial(behav_probs, 1).squeeze()
        elif policy == 'off_uniform':
            randm_draws = int(self.eps * len(action_probs))
            if randm_draws > 0:
                actions = action_probs.max(-1)[1].squeeze()
                action_loc = actions.float().mean()
                # randm_indices = torch.multinomial(torch.ones(len(action_probs)) / len(action_probs), randm_draws)
                randm_indices = torch.multinomial(waff_dis, randm_draws)
                actions[randm_indices] = torch.randint(0, self.args.n_actions, (randm_draws,)).to(device)
                behav_probs = action_probs.detach()
                behav_probs[randm_indices] = torch.Tensor(
                    [1 / self.args.n_actions for i in range(self.args.n_actions)]).to(device)
            else:
                actions = action_probs.max(-1)[1].squeeze()
                action_loc = actions.float().mean()
                behav_probs = action_probs.detach()
        elif policy == 'on':
            actions = action_probs.max(-1)[1].squeeze()
            action_loc = actions.float().mean()
            behav_probs = action_probs.detach()
        elif policy == 'q_val':
            actions = q.max(-1)[1].squeeze()
            action_loc = actions.float().mean()
            behav_probs = action_probs.detach()

        if writer is not None:
            writer.add_scalar("action/loc", action_loc, self.action_stats_count.value())
            self.action_stats_count.increment()
        # log_probs = torch.log(sel_behav_probs)
        # entropy = - (behav_probs * torch.log(behav_probs)).sum()
        return actions, behav_probs

    def fe_extr_warm_start(self, sp_feature_ext, writer=None):
        # dloader = DataLoader(MultiDiscSpGraphDsetBalanced(length=self.args.fe_warmup_iterations * 10), batch_size=10,
        #                      shuffle=True, pin_memory=True)
        dloader = DataLoader(MultiDiscSpGraphDset(length=self.args.fe_warmup_iterations * 10), batch_size=10,
                                shuffle=True, pin_memory=True)
        criterion = ContrastiveLoss(delta_var=0.5, delta_dist=1.5)
        optimizer = torch.optim.Adam(sp_feature_ext.parameters(), lr=2e-3)
        for i, (data, gt) in enumerate(dloader):
            data, gt = data.to(sp_feature_ext.device), gt.to(sp_feature_ext.device)
            pred = sp_feature_ext(data)

            l2_reg = None
            if self.args.l2_reg_params_weight != 0:
                for W in list(sp_feature_ext.parameters()):
                    if l2_reg is None:
                        l2_reg = W.norm(2)
                    else:
                        l2_reg = l2_reg + W.norm(2)
            if l2_reg is None:
                l2_reg = 0

            loss = criterion(pred, gt) + l2_reg * self.args.l2_reg_params_weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if writer is not None:
                writer.add_scalar("loss/fe_warm_start", loss.item(), self.writer_idx_warmup_loss)
                self.writer_idx_warmup_loss += 1

    def agent_forward(self, env, model, round_n, state=None, grad=True, post_input=False):
        with torch.set_grad_enabled(grad):
            if state is None:
                state = env.state
            inp = [obj.float().to(model.module.device) for obj in state + [env.raw + env.gt_seg.to(env.raw.device), 
                                                                           env.init_sp_seg]]
            return model(inp, sp_indices=env.sp_indices,
                         edge_index=env.edge_ids.to(model.module.device),
                         angles=env.edge_angles,
                         edge_features_1d=env.edge_features.to(model.module.device),
                         round_n=round_n, post_input=post_input)

    # Trains model
    def _step(self, memory, shared_model, env, optimizer, loss_weight, off_policy=True, writer=None):
        # starter code from https://github.com/Kaixhin/ACER/
        action_size = memory.memory[0].action.size(0)
        policy_loss, value_loss, cum_side_loss = 0, 0, 0
        l2_reg = None

        # Calculate n-step returns in forward view, stepping backwards from the last state
        t = len(memory)
        if t <= 1:
            return
        for state, action, reward, behav_policy_proba, done in reversed(memory.memory):
            t -= 1
            q, side_loss = self.agent_forward(env, shared_model, t, state)
            policy_proba = torch.softmax(q, -1)
            v = (q * policy_proba).sum(-1)
            if done and t == len(memory) - 1:
                q_ret_t = torch.zeros_like(env.state[0]).to(shared_model.module.device)
            elif t == len(memory) - 1:
                q_ret_t = v.detach()
                t -= 1
                continue  # here q_ret is for current step, need one more step for estimation
            # Importance sampling weights ρ ← π(∙|s_i) / µ(∙|s_i); 1 for on-policy
            if off_policy:
                rho = policy_proba.detach() / behav_policy_proba.detach()
            else:
                rho = torch.ones(1, action_size)

            # Qret ← r_i + γQret
            q_ret_t = reward + self.args.discount * q_ret_t

            rho_t = rho.gather(-1, action.unsqueeze(-1)).squeeze()

            # Value update dθ ← dθ - ∇θ∙1/2∙(Qret - Q(s_i, a_i; θ))^2
            q_t = q.gather(-1, action.unsqueeze(-1)).squeeze()
            value_loss = value_loss + ((q_ret_t - q_t) ** 2 / 2).mean()  # Least squares loss
            cum_side_loss = cum_side_loss + side_loss

            # Qret ← ρ¯_a_i∙(Qret - Q(s_i, a_i; θ)) + V(s_i; θ)
            q_ret_t = rho_t.clamp(max=self.args.trace_max) * (q_ret_t - q_t.detach()) + v.detach()

            if self.args.l2_reg_params_weight != 0:
                for W in list(shared_model.parameters()):
                    if l2_reg is None:
                        l2_reg = W.norm(2)
                    else:
                        l2_reg = l2_reg + W.norm(2)
            if l2_reg is None:
                l2_reg = 0

        if writer is not None:
            writer.add_scalar("loss/ql", value_loss.item(), self.global_writer_loss_count.value())
            self.global_writer_loss_count.increment()

        # Update networks
        loss = (self.args.v_loss_weight * value_loss
                + l2_reg * self.args.l2_reg_params_weight) \
               / len(memory) + cum_side_loss * 5
        torch.autograd.set_detect_anomaly(True)
        self._update_networks(loss, optimizer, shared_model, writer)

    # Acts and trains model
    def train(self, rank, start_time, return_dict):

        device = torch.device("cuda:" + str(rank))
        print('Running on device: ', device)
        torch.cuda.set_device(device)
        torch.set_default_tensor_type(torch.FloatTensor)

        writer = None
        if not self.args.cross_validate_hp:
            writer = SummaryWriter(logdir=os.path.join(self.save_dir, 'logs'))
            # posting parameters
            param_string = ""
            for k, v in vars(self.args).items():
                param_string += ' ' * 10 + k + ': ' + str(v) + '\n'
            writer.add_text("params", param_string)

        self.setup(rank, self.args.num_processes)
        transition = namedtuple('Transition', ('state', 'action', 'reward', 'behav_policy_proba', 'done'))
        memory = TransitionData(capacity=self.args.t_max, storage_object=transition)

        env = SpGcnEnv(self.args, device, writer=writer, writer_counter=self.global_writer_quality_count,
                       win_event_counter=self.global_win_event_count)
        # Create shared network
        model = GcnEdgeAngle1dQ(self.args.n_raw_channels, self.args.n_embedding_features,
                                self.args.n_edge_features, self.args.n_actions, device, writer=writer)

        if self.args.no_fe_extr_optim:
            for param in model.fe_ext.parameters():
                param.requires_grad = False

        model.cuda(device)
        shared_model = DDP(model, device_ids=[model.device])


        # dloader = DataLoader(MultiDiscSpGraphDsetBalanced(no_suppix=False, create=False), batch_size=1, shuffle=True, pin_memory=True,
        #                      num_workers=0)
        dloader = DataLoader(MultiDiscSpGraphDset(no_suppix=False), batch_size=1, shuffle=True, pin_memory=True,
                             num_workers=0)
        # Create optimizer for shared network parameters with shared statistics
        # optimizer = CstmAdam(shared_model.parameters(), lr=self.args.lr, betas=self.args.Adam_betas,
        #                      weight_decay=self.args.Adam_weight_decay)
        optimizer = Adam(shared_model.parameters(), lr=self.args.lr)

        if self.args.fe_extr_warmup and rank == 0 and not self.args.test_score_only:
            fe_extr = SpVecsUnet(self.args.n_raw_channels, self.args.n_embedding_features, device)
            fe_extr.cuda(device)
            self.fe_extr_warm_start(fe_extr, writer=writer)
            shared_model.module.fe_ext.load_state_dict(fe_extr.state_dict())
            if self.args.model_name == "" and not self.args.no_save:
                torch.save(shared_model.state_dict(), os.path.join(self.save_dir, 'agent_model'))
            elif not self.args.no_save:
                torch.save(shared_model.state_dict(), os.path.join(self.save_dir, self.args.model_name))

        dist.barrier()

        if self.args.model_name != "":
            shared_model.load_state_dict(torch.load(os.path.join(self.save_dir, self.args.model_name)))
        elif self.args.fe_extr_warmup:
            print('loaded fe extractor')
            shared_model.load_state_dict(torch.load(os.path.join(self.save_dir, 'agent_model')))

        if not self.args.test_score_only:
            quality = self.args.stop_qual_scaling + self.args.stop_qual_offset
            while self.global_count.value() <= self.args.T_max:
                if self.global_count.value() == 78:
                    a = 1
                self.update_env_data(env, dloader, device)
                # waff_dis = torch.softmax(env.edge_features[:, 0].squeeze() + 1e-30, dim=0)
                # waff_dis = torch.softmax(env.gt_edge_weights + 0.5, dim=0)
                waff_dis = torch.softmax(torch.ones_like(env.gt_edge_weights), dim=0)
                loss_weight = torch.softmax(env.gt_edge_weights + 1, dim=0)
                env.reset()
                state = [env.state[0].clone(), env.state[1].clone()]

                self.eps = self.eps_rule.apply(self.global_count.value(), quality)
                env.stop_quality = self.stop_qual_rule.apply(self.global_count.value(), quality)

                with open(os.path.join(self.save_dir, 'runtime_cfg.yaml')) as info:
                    args_dict = yaml.load(info, Loader=yaml.Loader)
                    if args_dict is not None:
                        if 'eps' in args_dict:
                            if self.args.eps != args_dict['eps']:
                                self.eps = args_dict['eps']
                        if 'safe_model' in args_dict:
                            self.args.safe_model = args_dict['safe_model']
                        if 'add_noise' in args_dict:
                            self.args.add_noise = args_dict['add_noise']

                if writer is not None:
                    writer.add_scalar("step/epsilon", self.eps, env.writer_counter.value())

                if self.args.safe_model and not self.args.no_save:
                    if rank == 0:
                        if self.args.model_name_dest != "":
                            torch.save(shared_model.state_dict(),
                                       os.path.join(self.save_dir, self.args.model_name_dest))
                        else:
                            torch.save(shared_model.state_dict(), os.path.join(self.save_dir, 'agent_model'))

                while not env.done:
                    # Calculate policy and values
                    post_input = True if (self.global_count.value() + 1) % 10 == 0 and env.counter == 0 else False
                    q, _ = self.agent_forward(env, shared_model, env.counter, grad=False, post_input=post_input)
                    # Step
                    action, behav_policy_proba = self.get_action(q, 'off_uniform', waff_dis, device, writer)
                    state_, reward, quality = env.execute_action(action)

                    if self.args.add_noise:
                        if self.global_count.value() > 110 and self.global_count.value() % 5:
                            noise = torch.randn_like(reward) * 0.8
                            reward = reward + noise

                    memory.push(state, action, reward, behav_policy_proba, env.done)

                    # Train the network
                    # self._step(memory, shared_model, env, optimizer, loss_weight, off_policy=True, writer=writer)

                    # reward = self.args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
                    # done = done or episode_length >= self.args.max_episode_length  # Stop episodes at a max length
                    state = state_

                self.global_count.increment()
                if "self_reg" in self.args.eps_rule and quality <= 2:
                    break

                while len(memory) > 0:
                    self._step(memory, shared_model, env, optimizer, loss_weight, off_policy=True, writer=writer)
                    memory.clear()

        dist.barrier()
        if rank == 0:
            if not self.args.cross_validate_hp and not self.args.test_score_only and not self.args.no_save:
                # pass
                if self.args.model_name_dest != "":
                    torch.save(shared_model.state_dict(), os.path.join(self.save_dir, self.args.model_name_dest))
                    print('saved')
                else:
                    torch.save(shared_model.state_dict(), os.path.join(self.save_dir, 'agent_model'))
            if self.args.cross_validate_hp or self.args.test_score_only:
                test_score = 0
                env.writer = None
                for i in range(20):
                    self.update_env_data(env, dloader, device)
                    env.reset()
                    self.eps = 0
                    env.stop_quality = 40
                    while not env.done:
                        # Calculate policy and values
                        q, _ = self.agent_forward(env, shared_model, env.counter, grad=False)
                        action, behav_policy_proba = self.get_action(policy_proba, q, v, policy='off_uniform',
                                                                     device=device, writer=writer)
                        _, _ = env.execute_action(action, self.global_count.value())

                    # import matplotlib.pyplot as plt;
                    # plt.imshow(env.get_current_soln());
                    # plt.show()
                    if env.win:
                        test_score += 1
                return_dict['test_score'] = test_score
                writer.add_text("time_needed", str((time.time() - start_time)))

        self.cleanup()
