# this agent implements the opposd algorithm introduced here : http://auai.org/uai2019/proceedings/papers/440.pdf
# using the trust region policy introduced by the acer algorithm: https://arxiv.org/pdf/1611.01224.pdf
import sys

sys.path.insert(0, '/g/kreshuk/hilt/projects/fewShotLearning/mu-net')
from models.sp_embed_unet import SpVecsUnet
from agents.distribution_correction import DensityRatio
from models.GCNNs.mc_glbl_edge_costs import GcnEdgeAngleConv1
import torch
import numpy as np
import os
import time
import torch.nn as nn
from agents.qlagent import QlAgent1
from mu_net.criteria.contrastive_loss import ContrastiveLoss
from torch.utils.data import DataLoader
from data.disjoint_discs import MultiDiscSpGraphDset
from mu_net.models.unet import UNet2d
from models.sp_embed_unet import SpVecsUnet
from agents.replayMemory import TransitionData
from environments.sp_grph_gcn_1 import SpGcnEnv
from models.GCNNs.mc_glbl_edge_costs import GcnEdgeAngle1dPQV
from utils.cstm_tensor import CstmTensor
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tensorboardX import SummaryWriter
import os
from optimizers.adam import CstmAdam
from utils.general import adjust_learning_rate
from collections import namedtuple
from agents.exploitation_functions import ActionPathTreeNodes, ExpSawtoothEpsDecay, NaiveDecay, GaussianDecay, Constant


class AgentOffpac(object):
    def __init__(self, args, shared_damped_model, global_count, global_writer_loss_count, global_writer_quality_count,
                 global_win_event_count, save_dir):
        super(AgentOffpac, self).__init__()
        self.args = args
        self.shared_damped_model = shared_damped_model
        self.global_count = global_count
        self.global_writer_loss_count = global_writer_loss_count
        self.global_writer_quality_count = global_writer_quality_count
        self.global_win_event_count = global_win_event_count
        self.writer_idx_warmup_loss = 0
        # self.eps = self.args.init_epsilon
        self.save_dir = save_dir
        if args.stop_qual_rule == 'naive':
            self.stop_qual_rule = NaiveDecay(initial_eps=args.init_stop_qual, episode_shrinkage=1,
                                             change_after_n_episodes=5)
        elif args.stop_qual_rule == 'gaussian':
            self.stop_qual_rule = GaussianDecay(args.stop_qual_final, args.stop_qual_scaling, args.stop_qual_offset,
                                                args.T_max)
        else:
            self.stop_qual_rule = NaiveDecay(args.init_stop_qual)

        if self.args.eps_rule == "treesearch":
            self.eps_rule = ActionPathTreeNodes()
        elif self.args.eps_rule == "sawtooth":
            self.eps_rule = ExpSawtoothEpsDecay()
        elif self.args.eps_rule == 'gaussian':
            self.eps_rule = GaussianDecay(args.eps_final, args.eps_scaling, args.eps_offset, args.T_max)
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
        os.environ['MASTER_PORT'] = '12355'
        # os.environ['GLOO_SOCKET_IFNAME'] = 'eno1'

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # Explicitly setting seed to make sure that models created in two processes
        # start from same random weights and biases.
        torch.manual_seed(self.args.seed)

    def cleanup(self):
        dist.destroy_process_group()

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
        if self.args.min_lr != 0:
            # Linearly decay learning rate
            new_lr = self.args.lr - ((self.args.lr - self.args.min_lr) *
                                     (1 - max((self.args.T_max - self.global_count.value()) / self.args.T_max, 1e-32)))
            adjust_learning_rate(optimizer, new_lr)

            if writer is not None:
                writer.add_scalar("loss/learning_rate", new_lr, self.global_writer_loss_count.value())

    def warm_start(self, transition_data):
        # warm-starting with data from initial behavior policy which is assumed to be uniform distribution
        qloss = self.get_qLoss(transition_data)
        ploss = 0
        for t in transition_data:
            ploss += self.policy.loss(self.policy(t.state), torch.ones(self.action_shape) / 2)
        ploss /= len(transition_data)
        self.q_val.optimizer.zero_grad()
        qloss.backward()
        for param in self.q_eval.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q_val.optimizer.step()
        self.policy.optimizer.zero_grad()
        for param in self.q_eval.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy.optimizer.step()

    def agent_forward(self, env, model, state=None, grad=True):
        with torch.set_grad_enabled(grad):
            if state is None:
                state = env.state
            return model([obj.float().to(model.module.device) for obj in state + [env.raw]], sp_indices=env.sp_indices,
                         edge_index=env.edge_ids.to(model.module.device),
                         angles=env.edge_angles.to(model.module.device),
                         edge_features_1d=env.edge_features.to(model.module.device))

    def get_action(self, action_probs, q, v, policy, device):
        if policy == 'off_sampled':
            behav_probs = action_probs.detach() + self.eps * (1 / self.args.n_actions - action_probs.detach())
            actions = torch.multinomial(behav_probs, 1).squeeze()
        elif policy == 'off_uniform':
            randm_draws = int(self.eps * len(action_probs))
            if randm_draws > 0:
                actions = action_probs.max(-1)[1].squeeze()
                randm_indices = torch.multinomial(torch.ones(len(action_probs)) / len(action_probs), randm_draws)
                actions[randm_indices] = torch.randint(0, self.args.n_actions, (randm_draws,)).to(device)
                behav_probs = action_probs.detach()
                behav_probs[randm_indices] = torch.Tensor(
                    [1 / self.args.n_actions for i in range(self.args.n_actions)]).to(device)
            else:
                actions = action_probs.max(-1)[1].squeeze()
                behav_probs = action_probs.detach()
        elif policy == 'on':
            actions = action_probs.max(-1)[1].squeeze()
            behav_probs = action_probs.detach()
        elif policy == 'q_val':
            actions = q.max(-1)[1].squeeze()
            behav_probs = action_probs.detach()

        # log_probs = torch.log(sel_behav_probs)
        # entropy = - (behav_probs * torch.log(behav_probs)).sum()
        return actions, behav_probs

    def fe_extr_warm_start(self, sp_feature_ext, writer=None):
        dataloader = DataLoader(MultiDiscSpGraphDset(length=100), batch_size=10,
                                shuffle=True, pin_memory=True)
        criterion = ContrastiveLoss(delta_var=0.5, delta_dist=1.5)
        optimizer = torch.optim.Adam(sp_feature_ext.parameters())
        for i, (data, gt) in enumerate(dataloader):
            data, gt = data.to(sp_feature_ext.device), gt.to(sp_feature_ext.device)
            pred = sp_feature_ext(data)
            loss = criterion(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if writer is not None:
                writer.add_scalar("loss/fe_warm_start", loss.item(), self.writer_idx_warmup_loss)
                self.writer_idx_warmup_loss += 1

    def _step(self, memory, shared_model, env, optimizer, off_policy=True, writer=None):
        if self.args.qnext_replace_cnt is not None and self.global_count.value() % self.args.qnext_replace_cnt == 0:
            self.shared_damped_model.load_state_dict(shared_model.state_dict())
        # according to thm3 in https://arxiv.org/pdf/1606.02647.pdf
        c_loss, a_loss = 0, 0
        transition_data = memory.memory
        correction = 0
        current = transition_data[0].time
        importance_weight = 1
        m = 0
        l2_reg = None
        # self.train_a2c = not self.train_a2c
        # self.opt_fe_extr = not self.train_a2c
        # self.dist_correction.update_density(transition_data, self.gamma)
        for i, t in enumerate(transition_data):
            if not t.terminal:
                _, q_, v_ = self.agent_forward(env, self.shared_damped_model, t.state_, False)
            pvals, q, v = self.agent_forward(env, shared_model, t.state)
            # pvals = nn.functional.softmax(qvals, -1).detach()  # this alternatively
            q_t = q.gather(-1, t.action.unsqueeze(-1)).squeeze()
            behav_policy_proba_t = t.behav_policy_proba.gather(-1, t.action.unsqueeze(-1)).squeeze().detach()
            pvals_t = pvals.gather(-1, t.action.unsqueeze(-1)).squeeze().detach()

            m = m + self.args.discount ** (t.time - current) * importance_weight
            importance_weight = importance_weight * self.args.lbd * \
                                torch.min(torch.ones(t.action.shape).to(shared_model.module.device),
                                          pvals_t / behav_policy_proba_t)

            if self.args.weight_l2_reg_params_weight != 0:
                for W in list(shared_model.parameters()):
                    if l2_reg is None:
                        l2_reg = W.norm(2)
                    else:
                        l2_reg = l2_reg + W.norm(2)
            if l2_reg is None:
                l2_reg = 0

            if t.terminal:
                c_loss = c_loss + nn.functional.mse_loss(t.reward * m, q_t * m)
            else:
                c_loss = c_loss + nn.functional.mse_loss((t.reward + self.args.discount * v_.detach()) * m, q_t * m)

        c_loss = c_loss / len(transition_data) + l2_reg * self.args.weight_l2_reg_params_weight

        # sample according to discounted state dis
        discount_distribution = [self.args.discount ** i for i in range(len(transition_data))]
        discount_distribution = np.exp(discount_distribution) / sum(np.exp(discount_distribution))  # softmax
        batch_ind = np.random.choice(len(transition_data), size=len(transition_data), p=discount_distribution)
        z = 0
        l2_reg = None
        for i in batch_ind:
            t = transition_data[i]
            # w = self.dist_correction.density_ratio(t.state.unsqueeze(0).unsqueeze(0).to(self.dist_correction.density_ratio.device)).detach().squeeze()
            w = 1
            z += w
            policy_proba, q, v = self.agent_forward(env, shared_model, t.state)

            policy_proba_t = policy_proba.gather(-1, t.action.unsqueeze(-1)).squeeze()
            q_t = q.gather(-1, t.action.unsqueeze(-1)).squeeze().detach()
            advantage_t = q_t - v.detach()
            behav_policy_proba_t = t.behav_policy_proba.gather(-1, t.action.unsqueeze(-1)).squeeze().detach()

            if self.args.weight_l2_reg_params_weight != 0:
                for W in list(shared_model.parameters()):
                    if l2_reg is None:
                        l2_reg = W.norm(2)
                    else:
                        l2_reg = l2_reg + W.norm(2)
            if l2_reg is None:
                l2_reg = 0

            a_loss = a_loss - (policy_proba_t.detach() / behav_policy_proba_t) * w * torch.log(
                policy_proba_t) * advantage_t
        z = z / len(batch_ind)
        a_loss = a_loss / z
        a_loss = a_loss / len(batch_ind)
        a_loss = a_loss / len(t.state)
        a_loss = torch.sum(a_loss) + l2_reg * self.args.weight_l2_reg_params_weight

        if writer is not None:
            writer.add_scalar("loss/critic", c_loss.item(), self.global_writer_loss_count.value())
            writer.add_scalar("loss/actor", a_loss.item(), self.global_writer_loss_count.value())
            print("c: ", c_loss.item())
            print("a: ", a_loss.item())
            self.global_writer_loss_count.increment()

        self._update_networks(a_loss + c_loss, optimizer, shared_model, writer)
        return

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

        transition = namedtuple('Transition',
                                ('state', 'action', 'reward', 'state_', 'behav_policy_proba', 'time', 'terminal'))
        memory = TransitionData(capacity=self.args.t_max, storage_object=transition)

        env = SpGcnEnv(self.args, device, writer=writer, writer_counter=self.global_writer_quality_count,
                       win_event_counter=self.global_win_event_count)
        dloader = DataLoader(MultiDiscSpGraphDset(no_suppix=False), batch_size=1, shuffle=True, pin_memory=True,
                             num_workers=0)
        # Create shared network
        model = GcnEdgeAngle1dPQV(self.args.n_raw_channels, self.args.n_embedding_features,
                                  self.args.n_edge_features, self.args.n_actions, device)
        model.cuda(device)
        shared_model = DDP(model, device_ids=[model.device])
        # Create optimizer for shared network parameters with shared statistics
        optimizer = CstmAdam(shared_model.parameters(), lr=self.args.lr, betas=self.args.Adam_betas,
                             weight_decay=self.args.Adam_weight_decay)

        if self.args.fe_extr_warmup and rank == 0:
            fe_extr = SpVecsUnet(self.args.n_raw_channels, self.args.n_embedding_features, device)
            fe_extr.cuda(device)
            self.fe_extr_warm_start(fe_extr, writer=writer)
            shared_model.module.fe_ext.load_state_dict(fe_extr.state_dict())
            if self.args.model_name == "":
                torch.save(fe_extr.state_dict(), os.path.join(self.save_dir, 'agent_model'))
            else:
                torch.save(shared_model.state_dict(), os.path.join(self.save_dir, self.args.model_name))
        dist.barrier()
        if self.args.model_name != "":
            shared_model.load_state_dict(torch.load(os.path.join(self.save_dir, self.args.model_name)))
        elif self.args.fe_extr_warmup:
            print('loaded fe extractor')
            shared_model.load_state_dict(torch.load(os.path.join(self.save_dir, 'agent_model')))

        self.shared_damped_model.load_state_dict(shared_model.state_dict())
        env.done = True  # Start new episode
        while self.global_count.value() <= self.args.T_max:
            if env.done:
                edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes, angles, affinities, gt = \
                    next(iter(dloader))
                edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes, angles, affinities, gt = \
                    edges.squeeze().to(device), edge_feat.squeeze()[:, 0:self.args.n_edge_features].to(
                        device), diff_to_gt.squeeze().to(device), \
                    gt_edge_weights.squeeze().to(device), node_labeling.squeeze().to(device), raw.squeeze().to(
                        device), nodes.squeeze().to(device), \
                    angles.squeeze().to(device), affinities.squeeze().numpy(), gt.squeeze()
                env.update_data(edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes,
                                angles, affinities, gt)
                env.reset()
                state = [env.state[0].clone(), env.state[1].clone()]
                episode_length = 0

                self.eps = self.eps_rule.apply(self.global_count.value())
                env.stop_quality = self.stop_qual_rule.apply(self.global_count.value())
                if writer is not None:
                    writer.add_scalar("step/epsilon", self.eps, env.writer_counter.value())

            while not env.done:
                # Calculate policy and values
                policy_proba, q, v = self.agent_forward(env, shared_model, grad=False)
                # average_policy_proba, _, _ = self.agent_forward(env, self.shared_average_model)
                # q_ret = v.detach()

                # Sample action
                # action = torch.multinomial(policy, 1)[0, 0]

                # Step
                action, behav_policy_proba = self.get_action(policy_proba, q, v, policy='off_uniform', device=device)
                state_, reward = env.execute_action(action, self.global_count.value())

                memory.push(state, action, reward.to(shared_model.module.device), state_, behav_policy_proba,
                            episode_length, env.done)

                # Train the network
                self._step(memory, shared_model, env, optimizer, off_policy=True, writer=writer)

                # reward = self.args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
                # done = done or episode_length >= self.args.max_episode_length  # Stop episodes at a max length
                episode_length += 1  # Increase episode counter
                state = state_

            # Break graph for last values calculated (used for targets, not directly as model outputs)
            self.global_count.increment()
            # Qret = 0 for terminal s

            while len(memory) > 0:
                self._step(memory, shared_model, env, optimizer, off_policy=True, writer=writer)
                memory.pop(0)

        dist.barrier()
        if rank == 0:
            if not self.args.cross_validate_hp:
                if self.args.model_name != "":
                    torch.save(shared_model.state_dict(), os.path.join(self.save_dir, self.args.model_name))
                else:
                    torch.save(shared_model.state_dict(), os.path.join(self.save_dir, 'agent_model'))
            else:
                test_score = 0
                env.writer = None
                for i in range(20):
                    self.update_env_data(env, dloader, device)
                    env.reset()
                    self.eps = 0
                    while not env.done:
                        # Calculate policy and values
                        policy_proba, q, v = self.agent_forward(env, shared_model, grad=False)
                        action, behav_policy_proba = self.get_action(policy_proba, q, v, policy='off_uniform',
                                                                     device=device)
                        _, _ = env.execute_action(action, self.global_count.value())
                    if env.win:
                        test_score += 1
                return_dict['test_score'] = test_score
                writer.add_text("time_needed", str((time.time() - start_time)))
        self.cleanup()
