from data.spg_dset import SpgDset
from data.disjoint_discs_balanced_graph import MultiDiscSpGraphDsetBalanced
import torch
from torch.optim import Adam
import time
from torch import nn
from models.sp_embed_unet import SpVecsUnet
from torch.nn import functional as F
from torch.utils.data import DataLoader
from agents.replayMemory import TransitionData_ts
from collections import namedtuple
from losses import ContrastiveLoss
from environments.batched_sp_sub_grph import SpGcnEnv
from models.GCNNs.sac_agent import GcnEdgeAC
from models.GCNNs.sac_agent1 import GcnEdgeAC_1
from models.GCNNs.batched_no_actor_mlp_sac_agent import GcnEdgeAC
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


class AgentSacTrainer_test_global(object):

    def __init__(self, cfg, args, global_count, global_writer_loss_count, global_writer_quality_count,
                 global_win_event_count, action_stats_count, save_dir):
        super(AgentSacTrainer_test_global, self).__init__()

        self.cfg = cfg
        self.args = args
        self.global_count = global_count
        self.global_writer_loss_count = global_writer_loss_count
        self.global_writer_quality_count = global_writer_quality_count
        self.global_win_event_count = global_win_event_count
        self.action_stats_count = action_stats_count
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
        raw, gt, indices = next(iter(dloader))
        raw, gt = raw.to(device), gt.to(device)
        edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling = dloader.dataset.get_graphs(indices, device)
        angles = None
        env.update_data(edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, angles, gt)

    def agent_forward(self, env, model, state, actions=None, grad=True, post_input=False, policy_opt=False):
        with torch.set_grad_enabled(grad):
            state_pixels, edge_ids, sp_indices, edge_angles, sub_graphs, counter, b_gt_edge_weights = self.state_to_cuda(state)
            if actions is not None:
                actions = actions.to(model.module.device)
            counter /= self.args.max_episode_length
            return model(state_pixels[:, 0, ...].unsqueeze(1),
                         edge_index=edge_ids,
                         angles=edge_angles,
                         round_n=counter,
                         actions=actions,
                         sp_indices=sp_indices,
                         gt_edges=b_gt_edge_weights,
                         sub_graphs=sub_graphs,
                         post_input=post_input,
                         policy_opt=policy_opt and grad)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @alpha.setter
    def alpha(self, value):
        self.log_alpha = torch.tensor(np.log(value)).to(self.device)
        self.log_alpha.requires_grad = True

    def update_critic(self, obs, action, reward, next_obs, not_done, env, model, optimizers):
        # dist = self.actor(next_obs)
        assert self.cfg.batch_size == 1, 'otw sum over batch'
        distribution, target_Q1, target_Q2, next_action = self.agent_forward(env, model, state=next_obs)
        target_Q1, target_Q2 = target_Q1.sum(), target_Q2.sum()
        log_prob = distribution.log_prob(next_action).sum()
        # target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob


        target_Q = reward.sum() + (not_done * self.cfg.discount * target_V)
        target_Q = target_Q.detach().squeeze()

        # get current Q estimates
        # current_Q1, current_Q2 = self.critic(obs, action)
        current_Q1, current_Q2 = self.agent_forward(env, model, state=obs, actions=action)
        # current_Q1, current_Q2 = current_Q1.sum(), current_Q2.sum()
        current_Q1, current_Q2 = current_Q1.sum(), current_Q2.sum()
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        optimizers.critic.zero_grad()
        critic_loss.backward()
        optimizers.critic.step()

        return critic_loss.item()

    def update_actor_and_alpha(self, obs, env, model, optimizers):
        # dist = self.actor(obs)
        distribution, actor_Q1, actor_Q2, action = self.agent_forward(env, model, state=obs, policy_opt=True)
        actor_Q1, actor_Q2 = actor_Q1.sum(), actor_Q2.sum()
        log_prob = distribution.log_prob(action).sum()

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        optimizers.actor.zero_grad()
        actor_loss.backward()
        optimizers.actor.step()

        alpha_loss = (self.alpha *
                      (-log_prob - action.shape[0]).detach()).mean()

        optimizers.temperature.zero_grad()
        alpha_loss.backward()
        optimizers.temperature.step()

        return actor_loss.item(), alpha_loss.item()

    def _step(self, replay_buffer, optimizers, env, model, step, writer=None):

        obs, action, reward, next_obs, done = replay_buffer.sample()
        not_done = int(not done)

        critic_loss = self.update_critic(obs, action, reward, next_obs, not_done, env, model, optimizers)

        if step % self.cfg.actor_update_frequency == 0:
            actor_loss, alpha_loss = self.update_actor_and_alpha(obs, env, model, optimizers)
            if writer is not None:
                writer.add_scalar("loss/actor", actor_loss, self.global_writer_loss_count.value())
                writer.add_scalar("loss/temperature", alpha_loss, self.global_writer_loss_count.value())

        if step % self.cfg.critic_target_update_frequency == 0:
            soft_update_params(model.module.critic, model.module.critic_tgt, self.critic_tau)

        if writer is not None:
            writer.add_scalar("loss/critic", critic_loss, self.global_writer_loss_count.value())

    # Acts and trains model
    def train(self, rank, start_time, return_dict):

        rns = torch.randint(0, 2 ** 32, torch.Size([10]))
        best_qual = - np.inf
        best_seed = None
        for i, rn in enumerate(rns):
            writer = SummaryWriter(logdir=os.path.join(self.save_dir, 'logs', str(i) + '_' + str(rn.item())))
            self.global_count.reset()
            self.global_writer_loss_count.reset()
            self.global_writer_quality_count.reset()
            self.global_win_event_count.reset()
            self.action_stats_count.reset()

            set_seed_everywhere(rn.item())
            qual = self.train_step(rank, start_time, return_dict, writer)
            if qual < best_qual:
                best_qual = qual
                best_seed = rns

        res = 'best seed is: ' + str(best_seed) + " with a qual of: " + str(best_qual)
        print(res)

        with open(os.path.join(self.save_dir, 'result.txt'), "w") as info:
            info.write(res)


    def train_step(self, rank, start_time, return_dict, writer):
        device = torch.device("cuda:" + str(rank))
        print('Running on device: ', device)
        torch.cuda.set_device(device)
        torch.set_default_tensor_type(torch.FloatTensor)

        self.setup(rank, self.args.num_processes)
        if self.cfg.MC_DQL:
            transition = namedtuple('Transition', ('episode'))
        else:
            transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
        memory = TransitionData_ts(capacity=self.args.t_max, storage_object=transition)

        env = SpGcnEnv(self.args, device, writer=writer, writer_counter=self.global_writer_quality_count,
                       win_event_counter=self.global_win_event_count)
        # Create shared network

        # model = GcnEdgeAC_1(self.cfg, self.args.n_raw_channels, self.args.n_embedding_features, 1, device, writer=writer)
        model = GcnEdgeAC(self.cfg, self.args, device, writer=writer)
        # model = GcnEdgeAC(self.cfg, self.args.n_raw_channels, self.args.n_embedding_features, 1, device, writer=writer)

        model.cuda(device)
        shared_model = DDP(model, device_ids=[model.device], find_unused_parameters=True)

        # dloader = DataLoader(MultiDiscSpGraphDsetBalanced(no_suppix=False, create=False), batch_size=1, shuffle=True, pin_memory=True,
        #                      num_workers=0)
        dloader = DataLoader(SpgDset(), batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True,
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
            # self.fe_extr_warm_start(fe_extr, writer=writer)
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
            best_quality = np.inf
            last_quals = []
            while self.global_count.value() <= self.args.T_max:
                if self.global_count.value() == 78:
                    a = 1
                self.update_env_data(env, dloader, device)
                # waff_dis = torch.softmax(env.edge_features[:, 0].squeeze() + 1e-30, dim=0)
                # waff_dis = torch.softmax(env.gt_edge_weights + 0.5, dim=0)
                waff_dis = torch.softmax(torch.ones_like(env.b_gt_edge_weights), dim=0)
                loss_weight = torch.softmax(env.b_gt_edge_weights + 1, dim=0)
                env.reset()
                # self.target_entropy = - float(env.gt_edge_weights.shape[0])
                self.target_entropy = -8.0

                env.stop_quality = self.stop_qual_rule.apply(self.global_count.value(), quality)
                if self.cfg.temperature_regulation == 'follow_quality':
                    self.alpha = self.eps_rule.apply(self.global_count.value(), quality)
                    print(self.alpha.item())

                with open(os.path.join(self.save_dir, 'runtime_cfg.yaml')) as info:
                    args_dict = yaml.full_load(info)
                    if args_dict is not None:
                        if 'safe_model' in args_dict:
                            self.args.safe_model = args_dict['safe_model']
                            args_dict['safe_model'] = False
                        if 'add_noise' in args_dict:
                            self.args.add_noise = args_dict['add_noise']
                        if 'critic_lr' in args_dict and args_dict['critic_lr'] != self.cfg.critic_lr:
                            self.cfg.critic_lr = args_dict['critic_lr']
                            adjust_learning_rate(critic_optimizer, self.cfg.critic_lr)
                        if 'actor_lr' in args_dict and args_dict['actor_lr'] != self.cfg.actor_lr:
                            self.cfg.actor_lr = args_dict['actor_lr']
                            adjust_learning_rate(actor_optimizer, self.cfg.actor_lr)
                        if 'alpha_lr' in args_dict and args_dict['alpha_lr'] != self.cfg.alpha_lr:
                            self.cfg.alpha_lr = args_dict['alpha_lr']
                            adjust_learning_rate(temp_optimizer, self.cfg.alpha_lr)
                with open(os.path.join(self.save_dir, 'runtime_cfg.yaml'), "w") as info:
                    yaml.dump(args_dict, info)

                if self.args.safe_model:
                    best_quality = quality
                    if rank == 0:
                        if self.args.model_name_dest != "":
                            torch.save(shared_model.state_dict(),
                                       os.path.join(self.save_dir, self.args.model_name_dest))
                        else:
                            torch.save(shared_model.state_dict(), os.path.join(self.save_dir, 'agent_model'))

                state = env.get_state()
                while not env.done:
                    # Calculate policy and values
                    post_input = True if (self.global_count.value() + 1) % 15 == 0 and env.counter == 0 else False
                    round_n = env.counter
                    # sample action for data collection
                    distr = None
                    if self.global_count.value() < self.cfg.num_seed_steps:
                        action = torch.rand_like(env.b_current_edge_weights)
                    else:
                        distr, _, _, action = self.agent_forward(env, shared_model, state=state, grad=False,
                                                             post_input=post_input)

                    logg_dict = {'temperature': self.alpha.item()}
                    if distr is not None:
                        logg_dict['mean_loc'] = distr.loc.mean().item()
                        logg_dict['mean_scale'] = distr.scale.mean().item()

                    if self.global_count.value() >= self.cfg.num_seed_steps and memory.is_full():
                        self._step(memory, optimizers, env, shared_model, self.global_count.value(), writer=writer)
                        self.global_writer_loss_count.increment()

                    next_state, reward, quality = env.execute_action(action, logg_dict)

                    last_quals.append(quality)
                    if len(last_quals) > 10:
                        last_quals.pop(0)

                    if self.args.add_noise:
                        noise = torch.randn_like(reward) * self.alpha.item()
                        reward = reward + noise

                    memory.push(self.state_to_cpu(state), action, reward, self.state_to_cpu(next_state), env.done)

                    # Train the network
                    # self._step(memory, shared_model, env, optimizer, loss_weight, off_policy=True, writer=writer)

                    # reward = self.args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
                    # done = done or episode_length >= self.args.max_episode_length  # Stop episodes at a max length
                    state = next_state

                self.global_count.increment()


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
        return sum(last_quals)/10

    def state_to_cpu(self, state):
        state_pixels, edge_ids, sp_indices, edge_angles, sub_graphs, counter, b_gt_edge_weights = state
        return state_pixels.cpu(), edge_ids.cpu(), sp_indices, edge_angles, sub_graphs.cpu(), counter, b_gt_edge_weights.cpu()

    def state_to_cuda(self, state):
        state_pixels, edge_ids, sp_indices, edge_angles, sub_graphs, counter, b_gt_edge_weights = state
        return state_pixels.to(self.device), edge_ids.to(self.device), sp_indices, \
               edge_angles, sub_graphs.to(self.device), counter, b_gt_edge_weights.to(self.device)