from data.disjoint_discs import MultiDiscSpGraphDset
import torch
from torch import nn
import time
from models.sp_embed_unet import SpVecsUnet
from torch.nn import functional as F
from torch.utils.data import DataLoader
from agents.replayMemory import TransitionData
from collections import namedtuple
from mu_net.criteria.contrastive_loss import ContrastiveLoss
from environments.sp_grph_gcn_1 import SpGcnEnv
from models.GCNNs.dueling_networks import GcnEdgeAngle1dPQA_dueling
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tensorboardX import SummaryWriter
import os
from optimizers.adam import CstmAdam
from scipy.stats import truncnorm
from utils.truncated_normal import TruncNorm
from agents.exploitation_functions import ActionPathTreeNodes, ExpSawtoothEpsDecay, NaiveDecay, GaussianDecay, Constant
import numpy as np
from utils.general import adjust_learning_rate
from torch.autograd import grad
from torch.autograd import Variable


class AgentAcerContinuousTrainer(object):

    def __init__(self, args, shared_average_model, global_count, global_writer_loss_count, global_writer_quality_count,
                 global_win_event_count, save_dir):
        super(AgentAcerContinuousTrainer, self).__init__()

        self.args = args
        self.shared_average_model = shared_average_model
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

        self.b_sigma_rule = GaussianDecay(args.b_sigma_final, args.b_sigma_scaling, args.p_sigma, args.T_max)

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

    def update_env_data(self, env, dloader, device):
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
        # nn.utils.clip_grad_norm_(shared_model.parameters(), self.args.max_gradient_norm)
        optimizer.step()
        if self.args.min_lr != 0:
            # Linearly decay learning rate
            new_lr = self.args.lr - ((self.args.lr - self.args.min_lr) *
                                     (1 - max((self.args.T_max - self.global_count.value()) / self.args.T_max, 1e-32)))
            adjust_learning_rate(optimizer, new_lr)

            if writer is not None:
                writer.add_scalar("loss/learning_rate", new_lr, self.global_writer_loss_count.value())

        # Update shared_average_model
        for shared_param, shared_average_param in zip(shared_model.parameters(),
                                                      self.shared_average_model.parameters()):
            shared_average_param.data = self.args.trust_region_decay * shared_average_param.data + (
                    1 - self.args.trust_region_decay) * shared_param.data

    # Computes an "efficient trust region" loss (policy head only) based on an existing loss and two distributions
    def _trust_region(self, g, k):
        # Compute dot products of gradients
        k_dot_g = (k * g).sum(0)
        k_dot_k = (k ** 2).sum(0)
        # Compute trust region update
        trust_factor = ((k_dot_g - self.args.trust_region_threshold) / k_dot_k).clamp(min=0)
        z = g - trust_factor * k
        return z

    def get_action(self, policy_means, p_dis, device, policy='off'):
        if policy == 'off':
            # use a truncated normal dis here https://en.wikipedia.org/wiki/Truncated_normal_distribution
            b_dis = TruncNorm(policy_means, self.b_sigma, 0, 1, self.args.density_eval_range)
            # rho is calculated as the distribution ration of the two normal distributions as described here:
            # https://www.researchgate.net/publication/257406150_On_the_existence_of_a_normal_approximation_to_the_distribution_of_the_ratio_of_two_independent_normal_random_variables
        elif policy == 'on':
            b_dis = p_dis
        else:
            assert False
        # sample actions alternatively consider unsampled approach by taking mean
        actions = b_dis.sample()
        # test = torch.stack([torch.from_numpy(actions).float().to(device),
        #              torch.from_numpy(policy_means).float().to(device)]).cpu().numpy()

        # print('sample sigma:', torch.sqrt(((actions - policy_means) ** 2).mean()).item())

        return actions, b_dis

    def fe_extr_warm_start(self, sp_feature_ext, writer=None):
        dataloader = DataLoader(MultiDiscSpGraphDset(length=10 * self.args.fe_warmup_iterations), batch_size=10,
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

    def agent_forward(self, env, model, action=None, state=None, grad=True, stats_only=False):
        with torch.set_grad_enabled(grad):
            if state is None:
                state = env.state
            inp = [obj.float().to(model.module.device) for obj in state + [env.raw, env.init_sp_seg]]
            return model(inp, action,
                         sp_indices=env.sp_indices,
                         edge_index=env.edge_ids.to(model.module.device),
                         angles=env.edge_angles.to(model.module.device),
                         edge_features_1d=env.edge_features.to(model.module.device),
                         stats_only=stats_only)

    # Trains model
    def _step(self, memory, shared_model, env, optimizer, off_policy=True, writer=None):
        torch.autograd.set_detect_anomaly(True)
        # starter code from https://github.com/Kaixhin/ACER/
        action_size = memory.memory[0].action.size(0)
        policy_loss, value_loss = 0, 0
        l2_reg = None

        # Calculate n-step returns in forward view, stepping backwards from the last state
        t = len(memory)
        if t <= 1:
            return
        for state, action, reward, b_dis, done in reversed(memory.memory):
            p, q, v, a, p_dis, sampled_action, q_prime = self.agent_forward(env, shared_model, action,
                                                                              state)
            average_p, average_action_rvs = self.agent_forward(env, self.shared_average_model, action,
                                                               state,
                                                               grad=False,
                                                               stats_only=True)

            if done and t == len(memory):
                q_ret = torch.zeros_like(env.state[0]).to(shared_model.module.device)
                q_opc = q_ret.clone()
            elif t == len(memory):
                q_ret = v.detach()
                q_opc = q_ret.clone()
                t -= 1
                continue  # here q_ret is for current step, need one more step for estimation

            if off_policy:
                # could also try relation of variances here
                rho = (p_dis.prob(action).detach()) \
                      / (b_dis.prob(action).detach())
                rho_prime = (p_dis.prob(sampled_action).detach()) \
                            / (b_dis.prob(sampled_action).detach())
                c = rho.pow(1/action_size).clamp(max=1)
                # c = rho.clamp(max=1)
            else:
                rho = torch.ones(1, action_size).to(shared_model.module.device)
                rho_prime = torch.ones(1, action_size).to(shared_model.module.device)

            # Qret ← r_i + γQret
            q_ret = reward + self.args.discount * q_ret
            q_opc = reward + self.args.discount * q_opc

            bias_weight = (1 - (self.args.trace_max / rho_prime)).clamp(min=0)

            # KL divergence k ← ∇θ0∙DKL[π(∙|s_i; θ_a) || π(∙|s_i; θ)]
            k = (p.detach() - average_p) / (self.args.p_sigma ** 4)
            g = rho.clamp(max=self.args.trace_max) * (q_opc - v.detach()) * p_dis.grad_pdf_mu(action).detach()
            if off_policy:
                g = g + bias_weight * (q_prime - v.detach()) * p_dis.grad_pdf_mu(sampled_action).detach()
            # Policy update dθ ← dθ + ∂θ/∂θ∙z*
            z_star = self._trust_region(g, k)
            tr_loss = (z_star * p * self.args.trust_region_weight).mean()

            # policy_loss = policy_loss - tr_loss

            # vanilla policy gradient with importance sampling
            lp = p_dis.log_prob(action)
            policy_loss = policy_loss - (c * lp * q.detach()).mean()

            # Value update dθ ← dθ - ∇θ∙1/2∙(Qret - Q(s_i, a_i; θ))^2
            value_loss = value_loss + (-(q_ret - q.detach()) * q).mean()  # Least squares loss
            value_loss = value_loss + (- (rho.clamp(max=1) * (q_ret - q.detach()) * v)).mean()

            # Qret ← ρ¯_a_i∙(Qret - Q(s_i, a_i; θ)) + V(s_i; θ)
            q_ret = c * (q_ret - q.detach()) + v.detach()
            q_opc = (q_ret - q.detach()) + v.detach()
            t -= 1

            if self.args.l2_reg_params_weight != 0:
                for W in list(shared_model.parameters()):
                    if l2_reg is None:
                        l2_reg = W.norm(2)
                    else:
                        l2_reg = l2_reg + W.norm(2)
            if l2_reg is None:
                l2_reg = 0

        if writer is not None:
            writer.add_scalar("loss/critic", value_loss.item(), self.global_writer_loss_count.value())
            writer.add_scalar("loss/actor", policy_loss.item(), self.global_writer_loss_count.value())
            self.global_writer_loss_count.increment()

        # Update networks
        loss = (self.args.p_loss_weight * policy_loss
                + self.args.v_loss_weight * value_loss
                + l2_reg * self.args.l2_reg_params_weight) / len(memory)

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
                       win_event_counter=self.global_win_event_count, discrete_action_space=False)
        # Create shared network
        model = GcnEdgeAngle1dPQA_dueling(self.args.n_raw_channels,
                                          self.args.n_embedding_features,
                                          self.args.n_edge_features, 1, self.args.exp_steps, self.args.p_sigma,
                                          device, self.args.density_eval_range)
        if self.args.no_fe_extr_optim:
            for param in model.fe_ext.parameters():
                param.requires_grad = False

        model.cuda(device)
        shared_model = DDP(model, device_ids=[model.device])
        dloader = DataLoader(MultiDiscSpGraphDset(no_suppix=False), batch_size=1, shuffle=True, pin_memory=True,
                             num_workers=0)
        # Create optimizer for shared network parameters with shared statistics
        optimizer = torch.optim.Adam(shared_model.parameters(), lr=self.args.lr, betas=self.args.Adam_betas,
                                     weight_decay=self.args.Adam_weight_decay)

        if self.args.fe_extr_warmup and rank == 0 and not self.args.test_score_only:
            fe_extr = SpVecsUnet(self.args.n_raw_channels, self.args.n_embedding_features, device)
            fe_extr.cuda(device)
            self.fe_extr_warm_start(fe_extr, writer=writer)
            shared_model.module.fe_ext.load_state_dict(fe_extr.state_dict())
            if self.args.model_name == "":
                torch.save(shared_model.state_dict(), os.path.join(self.save_dir, 'agent_model'))
            else:
                torch.save(shared_model.state_dict(), os.path.join(self.save_dir, self.args.model_name))

        dist.barrier()

        if self.args.model_name != "":
            shared_model.load_state_dict(torch.load(os.path.join(self.save_dir, self.args.model_name)))
        elif self.args.fe_extr_warmup:
            shared_model.load_state_dict(torch.load(os.path.join(self.save_dir, 'agent_model')))

        self.shared_average_model.load_state_dict(shared_model.state_dict())

        if not self.args.test_score_only:
            while self.global_count.value() <= self.args.T_max:
                if self.global_count.value() == 990:
                    a=1
                self.update_env_data(env, dloader, device)
                env.reset()
                state = [env.state[0].clone(), env.state[1].clone()]

                self.b_sigma = self.b_sigma_rule.apply(self.global_count.value())
                env.stop_quality = self.stop_qual_rule.apply(self.global_count.value())
                if writer is not None:
                    writer.add_scalar("step/b_variance", self.b_sigma, env.writer_counter.value())

                while not env.done:
                    # Calculate policy and values
                    policy_means, p_dis = self.agent_forward(env, shared_model, grad=False, stats_only=True)

                    # Step
                    action, b_rvs = self.get_action(policy_means, p_dis, device)
                    state_, reward = env.execute_action(action, self.global_count.value())

                    memory.push(state, action, reward, b_rvs, env.done)

                    # Train the network
                    # self._step(memory, shared_model, env, optimizer, off_policy=True, writer=writer)

                    # reward = self.args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
                    # done = done or episode_length >= self.args.max_episode_length  # Stop episodes at a max length
                    state = state_

                # Break graph for last values calculated (used for targets, not directly as model outputs)
                self.global_count.increment()

                self._step(memory, shared_model, env, optimizer, off_policy=True, writer=writer)
                memory.clear()
                # while len(memory) > 0:
                #     self._step(memory, shared_model, env, optimizer, off_policy=True, writer=writer)
                #     memory.pop(0)

        dist.barrier()
        if rank == 0:
            if not self.args.cross_validate_hp and not self.args.test_score_only and not self.args.no_save:
                # pass
                if self.args.model_name != "":
                    torch.save(shared_model.state_dict(), os.path.join(self.save_dir, self.args.model_name))
                    print('saved')
                else:
                    torch.save(shared_model.state_dict(), os.path.join(self.save_dir, 'agent_model'))
            if self.args.cross_validate_hp or self.args.test_score_only:
                test_score = 0
                env.writer = None
                for i in range(20):
                    self.update_env_data(env, dloader, device)
                    env.reset()
                    self.b_sigma = self.args.p_sigma
                    env.stop_quality = 40
                    while not env.done:
                        # Calculate policy and values
                        policy_means, p_dis = self.agent_forward(env, shared_model, grad=False, stats_only=True)
                        action, b_rvs = self.get_action(policy_means, p_dis, device)
                        _, _ = env.execute_action(action, self.global_count.value())

                    # import matplotlib.pyplot as plt;
                    # plt.imshow(env.get_current_soln());
                    # plt.show()
                    if env.win:
                        test_score += 1
                return_dict['test_score'] = test_score
                writer.add_text("time_needed", str((time.time() - start_time)))
        self.cleanup()
