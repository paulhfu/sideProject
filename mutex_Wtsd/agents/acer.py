from data.disjoint_discs import MultiDiscSpGraphDset
import torch
from torch import nn
from models.sp_embed_unet import SpVecsUnet
from torch.nn import functional as F
from torch.utils.data import DataLoader
from agents.replayMemory import TransitionData
from collections import namedtuple
from mu_net.criteria.contrastive_loss import ContrastiveLoss
from environments.sp_grph_gcn_1 import SpGcnEnv
from models.GCNNs.mc_glbl_edge_costs import GcnEdgeAngle1dPQV
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tensorboardX import SummaryWriter
import os
from optimizers.adam import CstmAdam
from agents.exploitation_functions import ActionPathTreeNodes, ExpSawtoothEpsDecay, NaiveDecay, GaussianDecay, Constant
import numpy as np
from utils.general import adjust_learning_rate

class AgentAcerTrainer(object):

    def __init__(self, args, shared_average_model, global_count, global_writer_loss_count, global_writer_quality_count,
                 global_win_event_count, save_dir):
        super(AgentAcerTrainer, self).__init__()

        self.args = args
        self.shared_average_model = shared_average_model
        self.global_count = global_count
        self.global_writer_loss_count = global_writer_loss_count
        self.global_writer_quality_count = global_writer_quality_count
        self.global_win_event_count = global_win_event_count
        self.writer_idx_warmup_loss = 0
        self.eps = self.args.init_epsilon
        self.save_dir = save_dir
        if args.stop_qual_rule == 'naive':
            self.stop_qual_rule = NaiveDecay(initial_eps=args.init_stop_qual, episode_shrinkage=1,
                                             change_after_n_episodes=5)
        elif args.stop_qual_rule == 'gaussian':
            self.stop_qual_rule = GaussianDecay(args.stop_qual_final, args.stop_qual_scaling, args.stop_qual_offset, args.T_max)
        else:
            self.stop_qual_rule = NaiveDecay(args.init_stop_qual)

        if self.args.eps_rule == "treesearch":
            self.eps_rule = ActionPathTreeNodes()
        elif self.args.eps_rule == "sawtooth":
            self.eps_rule = ExpSawtoothEpsDecay()
        elif self.args.eps_rule == 'gaussian':
            self.eps_rule = GaussianDecay(args.eps_final, 1, 0, args.T_max)
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
    def _update_networks(self, loss, optimizer, shared_model):
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
            new_lr = self.args.lr - (self.args.min_lr * (1 - max((self.args.T_max - self.global_count.value()) /
                                                                 self.args.T_max, 1e-32)))
            adjust_learning_rate(optimizer, new_lr)

        # Update shared_average_model
        for shared_param, shared_average_param in zip(shared_model.parameters(), self.shared_average_model.parameters()):
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
                behav_probs[randm_indices] = torch.Tensor([1/self.args.n_actions for i in range(self.args.n_actions)]).to(device)
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

    def agent_forward(self, env, model, state=None, grad=True):
        with torch.set_grad_enabled(grad):
            if state is None:
                state = env.state
            return model([obj.float().to(model.module.device) for obj in state + [env.raw]], sp_indices=env.sp_indices,
                         edge_index=env.edge_ids.to(model.module.device), angles=env.edge_angles.to(model.module.device),
                         edge_features_1d=env.edge_features.to(model.module.device))

    # Trains model
    def _step(self, memory, shared_model, env, optimizer, off_policy=True, writer=None):
        # starter code from https://github.com/Kaixhin/ACER/
        action_size = memory.memory[0].action.size(0)
        policy_loss, value_loss = 0, 0
        l2_reg = None

        # Calculate n-step returns in forward view, stepping backwards from the last state
        t = len(memory)
        for state, action, reward, behav_policy_proba, done in reversed(memory.memory):
            policy_proba, q, v = self.agent_forward(env, shared_model, state)
            average_policy_proba, _, _ = self.agent_forward(env, self.shared_average_model, state, grad=False)
            tr_loss, p_loss = 0, 0
            if done and t == len(memory):
                q_ret_t = torch.zeros_like(env.state[0]).to(shared_model.module.device)
            elif t == len(memory):
                q_ret_t = v.detach()
            # Importance sampling weights ρ ← π(∙|s_i) / µ(∙|s_i); 1 for on-policy
            if off_policy:
                rho = policy_proba.detach() / behav_policy_proba.detach()
            else:
                rho = torch.ones(1, action_size)

            # Qret ← r_i + γQret
            q_ret_t = reward + self.args.discount * q_ret_t
            # Advantage A ← Qret - V(s_i; θ)
            adv = q.detach() - v.unsqueeze(-1).detach()
            adv_ret_t = q_ret_t.detach() - v.detach()

            policy_proba_t = policy_proba.gather(-1, action.unsqueeze(-1)).squeeze()
            rho_t = rho.gather(-1, action.unsqueeze(-1)).squeeze()
            bias_weight = (1 - self.args.trace_max / rho).clamp(min=0)

            if not self.args.trust_region:
                # g ← min(c, ρ_a_i)∙∇θ∙log(π(a_i|s_i; θ))∙A
                p_loss = rho_t.clamp(max=self.args.trace_max) * policy_proba_t.log() * adv_ret_t
                # Off-policy bias correction
                if off_policy:
                    # g ← g + Σ_a [1 - c/ρ_a]_+∙π(a|s_i; θ)∙∇θ∙log(π(a|s_i; θ))∙(Q(s_i, a; θ) - V(s_i; θ)
                    p_loss = p_loss + ((bias_weight * policy_proba.log() * adv) * policy_proba.detach()).sum(-1)
                p_loss = p_loss.mean()
            if self.args.trust_region:
                # KL divergence k ← ∇θ0∙DKL[π(∙|s_i; θ_a) || π(∙|s_i; θ)]
                k = (- average_policy_proba / (policy_proba.detach() + 1e-10)).sum(-1)
                g = rho_t.clamp(max=self.args.trace_max) * adv_ret_t / policy_proba_t.detach()
                if off_policy:
                    g = g + (bias_weight * adv).sum(-1)
                # Policy update dθ ← dθ + ∂θ/∂θ∙z*
                z_star = self._trust_region(g, k)
                tr_loss = z_star * policy_proba_t * self.args.trust_region_weight

            # Entropy regularisation dθ ← dθ + β∙∇θH(π(s_i; θ))
            entropy_loss = -(self.args.entropy_weight * (policy_proba.log() * policy_proba).sum(-1))
            policy_loss = policy_loss - tr_loss.mean() - p_loss - entropy_loss.mean()

            # Value update dθ ← dθ - ∇θ∙1/2∙(Qret - Q(s_i, a_i; θ))^2
            q_t = q.gather(-1, action.unsqueeze(-1)).squeeze()
            value_loss = value_loss + ((q_ret_t - q_t) ** 2 / 2).mean()  # Least squares loss

            # Qret ← ρ¯_a_i∙(Qret - Q(s_i, a_i; θ)) + V(s_i; θ)
            q_ret_t = rho_t.clamp(max=1) * (q_ret_t - q_t.detach()) + v.detach()
            t -= 1

            if self.args.weight_l2_reg_params_weight != 0:
                for W in list(shared_model.parameters()):
                    if l2_reg is None:
                        l2_reg = W.norm(2)
                    else:
                        l2_reg = l2_reg + W.norm(2)
            if l2_reg is None:
                l2_reg = 0

        if writer is not None:
            writer.add_scalar("loss/critic", value_loss.item(), self.global_writer_loss_count.value())
            writer.add_scalar("loss/actor", value_loss.item(), self.global_writer_loss_count.value())
            self.global_writer_loss_count.increment()

        # Update networks
        loss = (policy_loss + value_loss + l2_reg * self.args.weight_l2_reg_params_weight) / len(memory)
        torch.autograd.set_detect_anomaly(True)
        self._update_networks(loss, optimizer, shared_model)

    # Acts and trains model
    def train(self, rank):

        device = torch.device("cuda:" + str(rank))
        print('Running on device: ', device)
        torch.cuda.set_device(device)
        torch.set_default_tensor_type(torch.FloatTensor)

        writer = SummaryWriter(logdir=os.path.join(self.save_dir, 'logs'))
        self.setup(rank, self.args.num_processes)
        transition = namedtuple('Transition', ('state', 'action', 'reward', 'behav_policy_proba', 'done'))
        memory = TransitionData(capacity=self.args.t_max, storage_object=transition)

        env = SpGcnEnv(self.args, device, writer=writer, writer_counter=self.global_writer_quality_count,
                       win_event_counter=self.global_win_event_count)
        # Create shared network
        model = GcnEdgeAngle1dPQV(self.args.n_raw_channels, self.args.n_embedding_features,
                                         self.args.n_edge_features, self.args.n_actions, device)
        model.cuda(device)
        shared_model = DDP(model, device_ids=[model.device])
        dloader = DataLoader(MultiDiscSpGraphDset(no_suppix=False), batch_size=1, shuffle=True, pin_memory=True,
                             num_workers=0)
        # Create optimizer for shared network parameters with shared statistics
        optimizer = CstmAdam(shared_model.parameters(), lr=self.args.lr, betas=self.args.Adam_betas,
                             weight_decay=self.args.Adam_weight_decay)

        if self.args.fe_extr_warmup and rank == 0:
            fe_extr = SpVecsUnet(self.args.n_raw_channels, self.args.n_embedding_features, device)
            fe_extr.cuda(device)
            self.fe_extr_warm_start(fe_extr, writer=writer)
            shared_model.module.fe_ext.load_state_dict(fe_extr.state_dict())
            torch.save(shared_model.state_dict(), os.path.join(self.save_dir, self.args.model_name))
        dist.barrier()
        if self.args.model_name != "":
            shared_model.load_state_dict(torch.load(os.path.join(self.save_dir, self.args.model_name)))

        self.shared_average_model.load_state_dict(shared_model.state_dict())
        env.done = True  # Start new episode

        while self.global_count.value() <= self.args.T_max:
            if env.done:
                edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes, angles, affinities, gt = \
                    next(iter(dloader))
                edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes, angles, affinities, gt = \
                    edges.squeeze().to(device), edge_feat.squeeze()[:, 0:self.args.n_edge_features].to(device), diff_to_gt.squeeze().to(device), \
                    gt_edge_weights.squeeze().to(device), node_labeling.squeeze().to(device), raw.squeeze().to(device), nodes.squeeze().to(device), \
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

                memory.push(state, action, reward, behav_policy_proba, env.done)

                # Train the network
                self._step(memory, shared_model, env, optimizer, off_policy=True, writer=writer)

                # reward = self.args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
                # done = done or episode_length >= self.args.max_episode_length  # Stop episodes at a max length
                episode_length += 1  # Increase episode counter
                state = state_


            # Break graph for last values calculated (used for targets, not directly as model outputs)
            self.global_count.increment()

            while len(memory) > 0:
                self._step(memory, shared_model, env, optimizer, off_policy=True, writer=writer)
                memory.pop(0)

        dist.barrier()
        if rank == 0:
            torch.save(shared_model.state_dict(), os.path.join(self.save_dir, 'agent_model'))
        self.cleanup()

