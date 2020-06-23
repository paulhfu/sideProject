from data.disjoint_discs import MultiDiscSpGraphDset
from data.disjoint_discs_balanced_graph import MultiDiscSpGraphDsetBalanced
import torch
from torch.optim import Adam
import time
from torch import nn
from models.sp_embed_unet import SpVecsUnetGcn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from agents.replayMemory import TransitionData_ts
from collections import namedtuple
from mu_net.criteria.contrastive_loss import ContrastiveLoss
from environments.sp_grph_gcn_2 import SpGcnEnv
from models.GCNNs.sac_agent import GcnEdgeAC
from models.GCNNs.sac_agent1 import GcnEdgeAC_1
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tensorboardX import SummaryWriter
import os
from optimizers.adam import CstmAdam
from agents.exploration_functions import ExponentialAverage, FollowLeadAvg, FollowLeadMin, RunningAverage, \
    ActionPathTreeNodes, ExpSawtoothEpsDecay, NaiveDecay, GaussianDecay, Constant
import numpy as np
from utils.general import adjust_learning_rate, soft_update_params, set_seed_everywhere, plt_bar_plot
import matplotlib.pyplot as plt
from utils.reward_functions import GraphDiceLoss
import yaml
import sys


class AgentSacSRTrainer(object):

    def __init__(self, cfg, args, global_count, global_writer_loss_count, global_writer_quality_count,
                 global_win_event_count, action_stats_count, save_dir):
        super(AgentSacSRTrainer, self).__init__()

        self.cfg = cfg
        self.args = args
        self.global_count = global_count
        self.global_writer_loss_count = global_writer_loss_count
        self.global_writer_quality_count = global_writer_quality_count
        self.global_win_event_count = global_win_event_count
        self.action_stats_count = action_stats_count
        self.writer_idx_warmup_loss = 0
        self.args.T_max = np.inf
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
            self.stop_qual_rule = Constant(args.stop_qual_final)

        if self.cfg.temperature_regulation == 'follow_quality':
            self.eps_rule = FollowLeadAvg(50 * (args.stop_qual_scaling + args.stop_qual_offset), 20, 0.1)
        elif self.cfg.temperature_regulation == 'constant':
            self.eps_rule = Constant(cfg.init_temperature)

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
        seed = torch.randint(0, 2 ** 32, torch.Size([5])).median()
        set_seed_everywhere(seed.item())

    def cleanup(self):
        dist.destroy_process_group()

    def update_env_data(self, env, dloader, device):
        edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes, affinities, gt = \
            next(iter(dloader))
        angles = None
        edges, edge_feat, gt_edge_weights, node_labeling, raw, nodes, affinities, gt = \
            edges.squeeze(), edge_feat.squeeze()[:, 0:self.args.n_edge_features], \
            gt_edge_weights.squeeze(), node_labeling.squeeze(), raw.squeeze(), nodes.squeeze(), \
            affinities.squeeze().numpy(), gt.squeeze()
        env.update_data(edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes,
                        angles, affinities, gt)

    def fe_extr_warm_start(self, sp_feature_ext, writer=None):
        # dloader = DataLoader(MultiDiscSpGraphDsetBalanced(length=self.args.fe_warmup_iterations * 10), batch_size=10,
        #                      shuffle=True, pin_memory=True)
        dloader = DataLoader(MultiDiscSpGraphDset(length=self.args.fe_warmup_iterations * 10,
                                                  less=True, no_suppix=False),
                             batch_size=1, shuffle=True, pin_memory=True)
        contrastive_l = ContrastiveLoss(delta_var=0.5, delta_dist=1.5)
        dice = GraphDiceLoss()
        small_lcf = nn.Sequential(
            nn.Linear(sp_feature_ext.n_embedding_channels, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 1024),
            nn.Linear(1024, 256),
            nn.Linear(256, 1),
        )
        small_lcf.cuda(device=sp_feature_ext.device)
        optimizer = torch.optim.Adam(sp_feature_ext.parameters(), lr=1e-3)
        for i, (data, node_labeling, gt_pix, gt_edges, edge_index) in enumerate(dloader):
            data, node_labeling, gt_pix, gt_edges, edge_index = data.to(sp_feature_ext.device), \
                                                                node_labeling.squeeze().to(sp_feature_ext.device), \
                                                                gt_pix.to(sp_feature_ext.device), \
                                                                gt_edges.squeeze().to(sp_feature_ext.device), \
                                                                edge_index.squeeze().to(sp_feature_ext.device)
            node_labeling = node_labeling.squeeze()
            stacked_superpixels = [node_labeling == n for n in node_labeling.unique()]
            sp_indices = [sp.nonzero() for sp in stacked_superpixels]

            edge_features, pred_embeddings, side_loss = sp_feature_ext(data, edge_index,
                                                                       torch.zeros_like(gt_edges, dtype=torch.float),
                                                                       sp_indices)

            pred_edge_weights = small_lcf(edge_features)

            l2_reg = None
            if self.args.l2_reg_params_weight != 0:
                for W in list(sp_feature_ext.parameters()):
                    if l2_reg is None:
                        l2_reg = W.norm(2)
                    else:
                        l2_reg = l2_reg + W.norm(2)
            if l2_reg is None:
                l2_reg = 0

            loss_pix = contrastive_l(pred_embeddings.unsqueeze(0), gt_pix)
            loss_edge = dice(pred_edge_weights.squeeze(), gt_edges.squeeze())
            loss = loss_pix + self.args.weight_edge_loss * loss_edge + \
                   self.args.weight_side_loss * side_loss + l2_reg * self.args.l2_reg_params_weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if writer is not None:
                writer.add_scalar("loss/fe_warm_start/ttl", loss.item(), self.writer_idx_warmup_loss)
                writer.add_scalar("loss/fe_warm_start/pix_embeddings", loss_pix.item(), self.writer_idx_warmup_loss)
                writer.add_scalar("loss/fe_warm_start/edge_embeddings", loss_edge.item(), self.writer_idx_warmup_loss)
                writer.add_scalar("loss/fe_warm_start/gcn_sideloss", side_loss.item(), self.writer_idx_warmup_loss)
                self.writer_idx_warmup_loss += 1

    def fe_extr_warm_start_1(self, sp_feature_ext, writer=None):
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

    def agent_forward(self, env, model, state, actions=None, grad=True, post_input=False, policy_opt=False):
        with torch.set_grad_enabled(grad):
            state_pixels, edge_ids, sp_indices, edge_angles, counter = state
            if actions is not None:
                actions = actions.to(model.module.device)
            counter /= self.args.max_episode_length
            return model(state_pixels[0].unsqueeze(0).unsqueeze(0),
                         edge_index=edge_ids.to(model.module.device),
                         angles=edge_angles,
                         round_n=counter,
                         actions=actions,
                         sp_indices=sp_indices,
                         gt_edges=env.gt_edge_weights,
                         post_input=post_input,
                         policy_opt=policy_opt and grad)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @alpha.setter
    def alpha(self, value):
        self.log_alpha = torch.tensor(np.log(value)).to(self.device)
        self.log_alpha.requires_grad = True

    def update_critic(self, obs, action, reward, next_obs, not_done, env, model):
        # dist = self.actor(next_obs)
        distribution, target_Q1, target_Q2, next_action = self.agent_forward(env, model, state=next_obs)
        # next_action = dist.rsample()
        log_prob = distribution.log_prob(next_action).sum()
        # target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2).sum() - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.cfg.discount * target_V)
        target_Q = target_Q.detach().squeeze()

        # get current Q estimates
        # current_Q1, current_Q2 = self.critic(obs, action)
        current_Q1, current_Q2 = self.agent_forward(env, model, state=obs, actions=action)
        current_Q1, current_Q2 = current_Q1.sum(), current_Q2.sum()
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        return critic_loss

    def update_actor_and_alpha(self, obs, env, model):
        # dist = self.actor(obs)
        distribution, actor_Q1, actor_Q2, action = self.agent_forward(env, model, state=obs, policy_opt=True)
        # action = dist.rsample()
        log_prob = distribution.log_prob(action).sum()
        # actor_Q1, actor_Q2 = self.critic(obs, action)
        # actor_Q1, actor_Q2 = self.agent_forward(state=obs, action=action, model=model)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()

        return actor_loss, alpha_loss

    def _step(self, replay_buffer, optimizers, env, model, step, writer=None):
        loss_critic, loss_actor, loss_alpha = 0, 0, 0
        batch = []
        for it in range(self.batch_size):
            batch.append(replay_buffer.sample())

        for obs, action, reward, next_obs, done in batch:
            not_done = int(not done)

            loss = self.update_critic(obs, action, reward, next_obs, not_done, env, model)
            loss_critic = loss_critic + loss

        loss_critic = loss_critic / self.batch_size
        optimizers.critic.zero_grad()
        loss_critic.backward()
        optimizers.critic.step()

        if step % self.cfg.actor_update_frequency == 0:
            for obs, action, reward, next_obs, done in batch:
                loss_a, loss_t = self.update_actor_and_alpha(obs, env, model)
                loss_actor = loss_actor + loss_a
                loss_alpha = loss_alpha + loss_t

        loss_actor = loss_actor / self.batch_size
        loss_alpha = loss_alpha / self.batch_size
        optimizers.actor.zero_grad()
        loss_actor.backward()
        optimizers.actor.step()
        optimizers.temperature.zero_grad()
        loss_alpha.backward()
        optimizers.temperature.step()

        if step % self.cfg.critic_target_update_frequency == 0:
            soft_update_params(model.module.critic, model.module.critic_tgt, self.critic_tau)

        if writer is not None:
            writer.add_scalar("loss/critic", loss_critic.item(), self.global_writer_loss_count.value())
            writer.add_scalar("loss/actor", loss_actor.item(), self.global_writer_loss_count.value())
            writer.add_scalar("loss/temperature", loss_alpha.item(), self.global_writer_loss_count.value())
            writer.add_scalar("value/temperature", self.alpha.detach().item(), self.global_writer_loss_count.value())

    def update_critic_episodic(self, episode, env, model):
        # dist = self.actor(next_obs)
        loss = 0
        target_Q = 0
        static_state = episode[0]
        steps = 0
        while episode:
            last_it = True
            for obs, action, reward, next_obs, done in reversed(episode[1:]):
                obs, next_obs = static_state + [obs], static_state + [next_obs]
                distribution, target_Q1, target_Q2, next_action = self.agent_forward(env, model, state=next_obs)
                # next_action = dist.rsample()
                log_prob = distribution.log_prob(next_action)
                try:
                    assert all(torch.isinf(log_prob) == False) and all(torch.isnan(log_prob) == False)
                except:
                    a = 1
                # target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
                if last_it:
                    last_it = False
                    if done:
                        target_Q = 0
                    else:
                        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
                        target_Q = self.cfg.discount * target_V
                target_Q = reward + target_Q
                target_Q = target_Q.detach()

                # get current Q estimates
                # current_Q1, current_Q2 = self.critic(obs, action)
                current_Q1, current_Q2 = self.agent_forward(env, model, state=obs, actions=action)

                # values = torch.stack(
                #     [current_Q1[:30].detach().cpu(), current_Q2[:30].detach().cpu(), target_Q[:30].detach().cpu()],
                #     dim=0).numpy()
                # fig = plt_bar_plot(values, labels=['CQ1', 'CQ2', 'TGTQ'])
                # plt.show()

                _loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
                loss = loss + _loss
                steps += 1

            episode = episode[:-1]
        return loss / steps

    def update_actor_and_alpha_episodic(self, episode, env, model):
        alpha_loss, actor_loss = 0, 0
        static_state = episode[0]
        steps = 0
        for obs, action, reward, next_obs, done in reversed(episode[1:]):
            obs, next_obs = static_state + [obs], static_state + [next_obs]
            distribution, actor_Q1, actor_Q2, action = self.agent_forward(env, model, state=obs, policy_opt=True)
            # action = dist.rsample()
            log_prob = distribution.log_prob(action).sum()
            try:
                assert all(torch.isinf(log_prob) == False) and all(torch.isnan(log_prob) == False)
            except:
                a = 1
            # actor_Q1, actor_Q2 = self.critic(obs, action)
            # actor_Q1, actor_Q2 = self.agent_forward(state=obs, action=action, model=model)

            actor_Q = torch.min(actor_Q1, actor_Q2)
            _actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

            if self.cfg.temperature_regulation == 'optimize':
                _alpha_loss = (self.alpha *
                               (-log_prob - self.target_entropy).detach()).mean()
                alpha_loss = alpha_loss + _alpha_loss
            actor_loss = actor_loss + _actor_loss
            steps += 1

        return actor_loss / steps, alpha_loss / steps

    def _step_episodic_mem(self, replay_buffer, optimizers, env, model, step, writer=None):
        loss_critic, loss_actor, loss_alpha = 0, 0, 0
        batch = []
        for it in range(self.batch_size):
            batch.append(replay_buffer.sample())

        for episode in batch:
            loss = self.update_critic_episodic(episode.episode, env, model)
            loss_critic = loss_critic + loss

        loss_critic = loss_critic / self.batch_size
        optimizers.critic.zero_grad()
        loss_critic.backward()
        optimizers.critic.step()

        if step % self.cfg.actor_update_frequency == 0:
            for episode in batch:
                loss_a, loss_t = self.update_actor_and_alpha_episodic(episode.episode, env, model)
                loss_actor = loss_actor + loss_a
                loss_alpha = loss_alpha + loss_t

            loss_actor = loss_actor / self.batch_size
            loss_alpha = loss_alpha / self.batch_size

            optimizers.actor.zero_grad()
            loss_actor.backward()
            optimizers.actor.step()
            if self.cfg.temperature_regulation == 'optimize':
                optimizers.temperature.zero_grad()
                loss_alpha.backward()
                optimizers.temperature.step()

            if step % self.cfg.critic_target_update_frequency == 0:
                soft_update_params(model.module.critic, model.module.critic_tgt, self.critic_tau)

        if writer is not None:
            writer.add_scalar("loss/critic", loss_critic.item(), self.global_writer_loss_count.value())
            writer.add_scalar("loss/actor", loss_actor.item(), self.global_writer_loss_count.value())
            if self.cfg.temperature_regulation == 'optimize':
                writer.add_scalar("loss/temperature", loss_alpha.item(), self.global_writer_loss_count.value())
            writer.add_scalar("value/temperature", self.alpha.detach().item(), self.global_writer_loss_count.value())

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
        if self.cfg.MC_DQL:
            transition = namedtuple('Transition', ('episode'))
        else:
            transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
        memory = TransitionData_ts(capacity=self.args.t_max, storage_object=transition)

        env = SpGcnEnv(self.args, device, writer=writer, writer_counter=self.global_writer_quality_count,
                       win_event_counter=self.global_win_event_count)
        # Create shared network

        model = GcnEdgeAC_1(self.cfg, self.args.n_raw_channels, self.args.n_embedding_features, 1, device, writer=writer)

        model.cuda(device)
        shared_model = DDP(model, device_ids=[model.device], find_unused_parameters=True)

        # dloader = DataLoader(MultiDiscSpGraphDsetBalanced(no_suppix=False, create=False), batch_size=1, shuffle=True, pin_memory=True,
        #                      num_workers=0)
        dloader = DataLoader(MultiDiscSpGraphDset(no_suppix=False), batch_size=1, shuffle=True, pin_memory=True,
                             num_workers=0)
        # Create optimizer for shared network parameters with shared statistics
        # optimizer = CstmAdam(shared_model.parameters(), lr=self.args.lr, betas=self.args.Adam_betas,
        #                      weight_decay=self.args.Adam_weight_decay)
        ######################
        self.action_range = 1
        self.device = torch.device(device)
        self.discount = 0.5
        self.critic_tau = self.cfg.critic_tau
        self.actor_update_frequency = self.cfg.actor_update_frequency
        self.critic_target_update_frequency = self.cfg.critic_target_update_frequency
        self.batch_size = self.cfg.batch_size

        self.log_alpha = torch.tensor(np.log(self.cfg.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        ######################
        # optimizers
        OptimizerContainer = namedtuple('OptimizerContainer', ('actor', 'critic', 'temperature'))
        actor_optimizer = torch.optim.Adam(shared_model.module.actor.parameters(),
                                           lr=self.cfg.actor_lr,
                                           betas=self.cfg.actor_betas)

        critic_optimizer = torch.optim.Adam(shared_model.module.critic.parameters(),
                                            lr=self.cfg.critic_lr,
                                            betas=self.cfg.critic_betas)

        temp_optimizer = torch.optim.Adam([self.log_alpha],
                                          lr=self.cfg.alpha_lr,
                                          betas=self.cfg.alpha_betas)

        optimizers = OptimizerContainer(actor_optimizer, critic_optimizer, temp_optimizer)

        if self.args.fe_extr_warmup and rank == 0 and not self.args.test_score_only:
            fe_extr = shared_model.module.fe_ext
            fe_extr.cuda(device)
            self.fe_extr_warm_start_1(fe_extr, writer=writer)
            if self.args.model_name == "" and not self.args.no_save:
                torch.save(fe_extr.state_dict(), os.path.join(self.save_dir, 'agent_model_fe_extr'))
            elif not self.args.no_save:
                torch.save(fe_extr.state_dict(), os.path.join(self.save_dir, self.args.model_name))

        dist.barrier()
        for param in model.fe_ext.parameters():
            param.requires_grad = False

        if self.args.model_name != "":
            shared_model.load_state_dict(torch.load(os.path.join(self.save_dir, self.args.model_name)))
        elif self.args.model_fe_name != "":
            shared_model.module.fe_ext.load_state_dict(torch.load(os.path.join(self.save_dir, self.args.model_fe_name)))
        elif self.args.fe_extr_warmup:
            print('loaded fe extractor')
            shared_model.module.fe_ext.load_state_dict(torch.load(os.path.join(self.save_dir, 'agent_model_fe_extr')))

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
                self.target_entropy = - float(env.gt_edge_weights.shape[0])

                env.stop_quality = self.stop_qual_rule.apply(self.global_count.value(), quality)
                if self.cfg.temperature_regulation == 'follow_quality':
                    self.alpha = self.eps_rule.apply(self.global_count.value(), quality)
                    print(self.alpha.item())

                with open(os.path.join(self.save_dir, 'runtime_cfg.yaml')) as info:
                    args_dict = yaml.full_load(info)
                    if args_dict is not None:
                        if 'safe_model' in args_dict:
                            self.args.safe_model = args_dict['safe_model']
                        if 'critic_lr' in args_dict and args_dict['critic_lr'] != self.cfg.critic_lr:
                            self.cfg.critic_lr = args_dict['critic_lr']
                            adjust_learning_rate(critic_optimizer, self.cfg.critic_lr)
                        if 'actor_lr' in args_dict and args_dict['actor_lr'] != self.cfg.actor_lr:
                            self.cfg.actor_lr = args_dict['actor_lr']
                            adjust_learning_rate(actor_optimizer, self.cfg.actor_lr)
                        if 'alpha_lr' in args_dict and args_dict['alpha_lr'] != self.cfg.alpha_lr:
                            self.cfg.alpha_lr = args_dict['alpha_lr']
                            adjust_learning_rate(temp_optimizer, self.cfg.alpha_lr)

                if self.args.safe_model and not self.args.no_save:
                    if rank == 0:
                        if self.args.model_name_dest != "":
                            torch.save(shared_model.state_dict(),
                                       os.path.join(self.save_dir, self.args.model_name_dest))
                        else:
                            torch.save(shared_model.state_dict(), os.path.join(self.save_dir, 'agent_model'))
                if self.cfg.MC_DQL:
                    state_pixels, edge_ids, sp_indices, edge_angles, counter = env.get_state()
                    state_ep = [state_pixels, edge_ids, sp_indices, edge_angles]
                    episode = [state_ep]
                    state = counter
                    while not env.done:
                        # Calculate policy and values
                        post_input = True if (self.global_count.value() + 1) % 15 == 0 and env.counter == 0 else False
                        round_n = env.counter
                        # sample action for data collection
                        if self.global_count.value() < self.cfg.num_seed_steps:
                            action = torch.rand_like(env.current_edge_weights)
                        else:
                            _, _, _, action = self.agent_forward(env, shared_model, state=state_ep + [state],
                                                                 grad=False, post_input=post_input)

                        action = action.cpu()

                        (_, _, _, _, next_state), reward, quality = env.execute_action(action)

                        episode.append((state, action, reward, next_state, env.done))
                        state = next_state

                    memory.push(episode)
                    if self.global_count.value() >= self.cfg.num_seed_steps and memory.is_full():
                        self._step_episodic_mem(memory, optimizers, env, shared_model, self.global_count.value(),
                                                writer=writer)
                        self.global_writer_loss_count.increment()
                else:
                    state = env.get_state()
                    while not env.done:
                        # Calculate policy and values
                        post_input = True if (self.global_count.value() + 1) % 15 == 0 and env.counter == 0 else False
                        round_n = env.counter
                        # sample action for data collection
                        if self.global_count.value() < self.cfg.num_seed_steps:
                            action = torch.rand_like(env.current_edge_weights)
                        else:
                            _, _, _, action = self.agent_forward(env, shared_model, state=state, grad=False,
                                                                 post_input=post_input)

                        action = action.cpu()
                        if self.global_count.value() >= self.cfg.num_seed_steps and memory.is_full():
                            self._step(memory, optimizers, env, shared_model, self.global_count.value(), writer=writer)
                            self.global_writer_loss_count.increment()

                        next_state, reward, quality = env.execute_action(action)

                        memory.push(state, action, reward, next_state, env.done)

                        # Train the network
                        # self._step(memory, shared_model, env, optimizer, loss_weight, off_policy=True, writer=writer)

                        # reward = self.args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
                        # done = done or episode_length >= self.args.max_episode_length  # Stop episodes at a max length
                        state = next_state

                self.global_count.increment()
                if "self_reg" in self.args.eps_rule and quality <= 2:
                    break

        dist.barrier()
        if rank == 0:
            if not self.args.cross_validate_hp and not self.args.test_score_only and not self.args.no_save:
                # pass
                if self.args.model_name_dest != "":
                    torch.save(shared_model.state_dict(), os.path.join(self.save_dir, self.args.model_name_dest))
                    print('saved')
                else:
                    torch.save(shared_model.state_dict(), os.path.join(self.save_dir, 'agent_model'))

        self.cleanup()
