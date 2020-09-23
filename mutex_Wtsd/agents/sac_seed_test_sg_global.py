from data.spg_dset import SpgDset
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
from environments.batched_sp_sub_grph import SpGcnEnv
from models.GCNNs.sac_agent1 import GcnEdgeAC_1
from models.GCNNs.batched_no_actor_mlp_qgpool_sac_agent import GcnEdgeAC
# from models.GCNNs.batched_no_actor_mlp_sac_agent import GcnEdgeAC
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils.sigmoid_normal1 import SigmNorm
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
import portalocker
import sys
from shutil import copyfile
from scipy.sparse.csgraph import dijkstra
from mu_net.criteria import ContrastiveLoss
from losses.rag_contrastive_triplet_loss import ContrastiveTripletLoss


class AgentSacTrainer_test_sg_global(object):

    def __init__(self, cfg, args, global_count, global_writer_loss_count, global_writer_quality_count,
                 global_win_event_count, action_stats_count, global_writer_count, save_dir):
        super(AgentSacTrainer_test_sg_global, self).__init__()

        self.cfg = cfg
        self.args = args
        self.global_count = global_count
        self.global_writer_loss_count = global_writer_loss_count
        self.global_writer_quality_count = global_writer_quality_count
        self.global_win_event_count = global_win_event_count
        self.action_stats_count = action_stats_count
        self.global_writer_count = global_writer_count
        self.contr_trpl_loss = ContrastiveTripletLoss(delta_var=0.5)
        self.contr_loss = ContrastiveLoss(delta_var=0.5, delta_dist=1.5)

        self.memory = TransitionData_ts(capacity=self.args.t_max)
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
            self.beta_rule = FollowLeadAvg(1, 80, 1)
        elif self.cfg.temperature_regulation == 'constant':
            self.eps_rule = Constant(cfg.init_temperature)

    def setup(self, rank, world_size, device):
        # BLAS setup
        os.environ['OMP_NUM_THREADS'] = '10'
        os.environ['MKL_NUM_THREADS'] = '10'

        # assert torch.cuda.device_count() == 1
        torch.set_default_tensor_type('torch.FloatTensor')

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = self.args.master_port

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def pretrain_embeddings(self, model, device, writer=None):
        dset = SpgDset(root_dir=self.args.data_dir)
        dloader = DataLoader(dset, batch_size=self.args.fe_warmup_batch_size, shuffle=True, pin_memory=True, num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        max_p = torch.nn.MaxPool2d(3, padding=1, stride=1)

        for i in range(self.args.fe_warmup_iterations):
            for it, (raw, gt, indices) in enumerate(dloader):
                raw, gt = raw.to(device), gt.to(device)
                edges, edge_feat, diff_to_gt, gt_edge_weights, sp_seg = dloader.dataset.get_graphs(indices, device)
                sp_seg_edge = torch.cat([(-max_p(-sp_seg) != sp_seg).float(), (max_p(sp_seg) != sp_seg).float()], 1)
                embeddings = model(torch.cat([raw, sp_seg_edge], 1))
                loss = self.contr_loss(embeddings, sp_seg.long().squeeze(1))

                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()

                if writer is not None:
                    writer.add_scalar("loss/fe_warm_start/ttl", loss.item(), it)
                del loss
                del embeddings
                break
            break
        a=1

    def cleanup(self):
        dist.destroy_process_group()

    def update_env_data(self, env, dloader, device):
        raw, gt, indices = next(iter(dloader))
        raw, gt = raw.to(device), gt.to(device)
        edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling = dloader.dataset.get_graphs(indices, device)
        angles = None
        env.update_data(edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, angles, gt)

    def agent_forward(self, env, model, state, actions=None, grad=True, post_input=False, post_model=False, policy_opt=False, embeddings_opt=False):
        with torch.set_grad_enabled(grad):
            raw, sp_seg, edge_ids, sp_indices, edge_angles, sub_graphs, sep_subgraphs, counter, b_gt_edge_weights, edge_offsets = self.state_to_cuda(state, env.device)
            if actions is not None:
                actions = actions.to(model.module.device)
            counter /= self.args.max_episode_length
            # model.module.writer.add_graph(model.module, (raw, b_gt_edge_weights, sp_indices, edge_ids, edge_angles, counter, sub_graphs, sep_subgraphs, actions, post_input, policy_opt), verbose=False)

            ret = model(raw,
                        sp_seg,
                        edge_index=edge_ids,
                        angles=edge_angles,
                        round_n=counter,
                        actions=actions,
                        sp_indices=sp_indices,
                        gt_edges=b_gt_edge_weights,
                        sub_graphs=sub_graphs,
                        sep_subgraphs=sep_subgraphs,
                        post_input=post_input,
                        policy_opt=policy_opt and grad,
                        embeddings_opt=embeddings_opt)

            if post_model and grad:
                for name, value in model.module.actor.named_parameters():
                    model.writer.add_histogram(name, value.data.cpu().numpy(), self.global_count.value())
                    model.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.global_count.value())
                for name, value in model.module.critic_tgt.named_parameters():
                    model.writer.add_histogram(name, value.data.cpu().numpy(), self.global_count.value())
                    model.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.global_count.value())

        return ret

    def update_critic(self, obs, action, reward, next_obs, not_done, env, model, optimizers):
        distribution, target_Q1, target_Q2, next_action, _ = self.agent_forward(env, model, state=next_obs)

        log_prob = distribution.log_prob(next_action)
        log_prob = log_prob[next_obs[5]].view(-1, self.args.s_subgraph).sum(-1)

        target_V = torch.min(target_Q1, target_Q2) - self.beta * log_prob
        # target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob

        # target_V = target_V[obs[4]].view(-1, self.args.s_subgraph).sum(-1)

        target_Q = reward + (not_done * self.cfg.discount * target_V)
        target_Q = target_Q.detach().squeeze()

        current_Q1, current_Q2 = self.agent_forward(env, model, state=obs, actions=action)

        # current_Q1 = current_Q1[obs[4]].view(-1, self.args.s_subgraph).sum(-1)
        # current_Q2 = current_Q2[obs[4]].view(-1, self.args.s_subgraph).sum(-1)
        critic_loss = (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)) / 2

        # loss = embedding_regularizer + critic_loss

        optimizers.critic.zero_grad()
        critic_loss.backward()
        optimizers.critic.step()

        return critic_loss.item()

    def _get_connected_paths(self, edges, weights, size, get_repulsive=False):
        graph = np.ones((size, size)) * np.inf
        graph[edges[0], edges[1]] = weights
        graph[edges[1], edges[0]] = weights
        dists = dijkstra(graph, directed=False)
        if get_repulsive:
            tril = np.tril(np.ones_like(dists), 0).astype(np.bool)
            dists[tril] = 0
            repulsive_edges = np.nonzero(dists == np.inf)
            if repulsive_edges[0].size > 0:
                return torch.stack([torch.from_numpy(repulsive_edges[0]), torch.from_numpy(repulsive_edges[1])], 0)
            return None
        tril = np.tril(np.ones_like(dists) * np.inf, 0)
        dists += tril
        attr_edges = np.nonzero(dists < np.inf)
        if attr_edges[0].size > 0:
            return torch.stack([torch.from_numpy(attr_edges[0]), torch.from_numpy(attr_edges[1])], 0)
        return None

    def get_embed_loss_contr_trpl(self, weights, obs, embeddings):
        b_attr_edges = []
        b_rep_edges = []
        for i in range(obs[1].shape[0]):
            edges = obs[2][:, obs[-1][i]:obs[-1][i+1]]
            edges = edges - edges.min()

            attr_weight_del = weights[obs[-1][i]:obs[-1][i + 1]] < self.cfg.weight_tolerance_attr
            attr_weights = weights[obs[-1][i]:obs[-1][i + 1]][attr_weight_del]
            if len(attr_weights) != 0:
                attr_weights = attr_weights - attr_weights.min()
                max = attr_weights.max()
                max = max if max != 0 else 1e-16
                attr_weights = attr_weights / max
                attr_weights += 1e-16  # make sure all edges exist in graph
                direct_attr = edges[:, attr_weight_del].numpy()
                b_attr_edges.append(self._get_connected_paths(direct_attr, attr_weights, edges.max()+1))
            else:
                b_attr_edges.append(None)

            rep_weight_del = weights[obs[-1][i]:obs[-1][i + 1]] <= self.cfg.weight_tolerance_rep
            rep_weights = weights[obs[-1][i]:obs[-1][i + 1]][rep_weight_del]
            if len(rep_weights) != 0:
                rep_weights = rep_weights - rep_weights.min()
                max = rep_weights.max()
                max = max if max != 0 else 1e-16
                rep_weights = rep_weights / max
                rep_weights += 1e-16  # make sure all edges exist in graph
                direct_rep = edges[:, rep_weight_del].numpy()
                b_rep_edges.append(self._get_connected_paths(direct_rep, rep_weights, edges.max()+1, get_repulsive=True))
            else:
                b_rep_edges.append(None)

        return self.contr_trpl_loss(
            embeddings, obs[1].long().to(embeddings.device), (b_attr_edges, b_rep_edges))

    def get_embed_loss_contr(self, weights, env, embeddings):
        segs = env.get_current_soln(weights)
        return self.contr_loss(embeddings, segs.long().to(embeddings.device))

    def update_embeddings(self, obs, env, model, optimizers):
        distribution, actor_Q1, actor_Q2, action, embeddings = self.agent_forward(env, model, grad=False, state=obs, policy_opt=False, embeddings_opt=True)

        # embedding_regularizer = torch.tensor(0.)
        weights = distribution.loc.detach()

        # weights = torch.autograd.grad(outputs=actor_loss, inputs=distribution.loc, retain_graph=True, create_graph=True, only_inputs=True)[0]
        # weights = weights.detach().cpu().numpy()
        weights = weights.cpu().numpy()

        loss = self.get_embed_loss_contr(weights, env, embeddings)

        optimizers.embeddings.zero_grad()
        loss.backward()
        optimizers.embeddings.step()
        return loss.item()

    def update_actor_and_alpha(self, obs, env, model, optimizers, embeddings_opt, writer=None):
        distribution, actor_Q1, actor_Q2, action = self.agent_forward(env, model, state=obs, policy_opt=True)

        log_prob = distribution.log_prob(action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        # actor_Q = actor_Q[obs[5]].view(-1, self.args.s_subgraph).sum(-1)
        log_prob = log_prob[obs[5]].view(-1, self.args.s_subgraph).sum(-1)

        actor_loss = (model.module.alpha.detach() * log_prob - actor_Q).mean()
        # reg_loss = - self.cfg.reg_scaler * torch.exp(-embedding_regularizer)

        loss = actor_loss

        optimizers.actor.zero_grad()
        loss.backward()
        optimizers.actor.step()

        alpha_loss = (model.module.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()

        optimizers.temperature.zero_grad()
        alpha_loss.backward()
        optimizers.temperature.step()

        return actor_loss.item(), alpha_loss.item()

    def _step(self, replay_buffer, optimizers, env, model, step, writer=None):

        (obs, action, reward, next_obs, done), sample_idx = replay_buffer.sample()
        not_done = int(not done)

        if not self.args.no_fe_extr_optim:
            if max(1, self.global_count.value()-self.args.t_max) % self.cfg.embeddings_update_frequency <= self.args.n_processes_per_gpu*self.args.n_gpu:
                embedd_loss = self.update_embeddings(obs, env, model, optimizers)
                if writer is not None:
                    writer.add_scalar("loss/embedd", embedd_loss, self.global_writer_loss_count.value())
                return

        critic_loss = self.update_critic(obs, action, reward, next_obs, not_done, env, model, optimizers)
        replay_buffer.report_sample_loss(critic_loss + reward.mean(), sample_idx)

        if step % self.cfg.actor_update_frequency == 0:
            actor_loss, alpha_loss = self.update_actor_and_alpha(obs, env, model, optimizers, writer)
            if writer is not None:
                writer.add_scalar("loss/actor", actor_loss, self.global_writer_loss_count.value())
                writer.add_scalar("loss/temperature", alpha_loss, self.global_writer_loss_count.value())

        if step % self.cfg.critic_target_update_frequency == 0:
            soft_update_params(model.module.critic, model.module.critic_tgt, self.cfg.critic_tau)

        if writer is not None:
            writer.add_scalar("loss/critic", critic_loss, self.global_writer_loss_count.value())

    # Acts and trains model
    def train(self, rank, start_time, return_dict, rn):

        self.log_dir = os.path.join(self.save_dir, 'logs', '_' + str(rn))
        writer = None
        if rank == 0:
            writer = SummaryWriter(logdir=self.log_dir)
            copyfile(os.path.join(self.save_dir, 'runtime_cfg.yaml'),
                     os.path.join(self.log_dir, 'runtime_cfg.yaml'))

            self.global_count.reset()
            self.global_writer_loss_count.reset()
            self.global_writer_quality_count.reset()
            self.global_win_event_count.reset()
            self.action_stats_count.reset()
            self.global_writer_count.reset()
        
        set_seed_everywhere(rn)
        if rank == 0:
            print('training with seed: ' + str(rn))
        score = self.train_step(rank, writer)
        if rank == 0:
            return_dict['score'] = score
            del self.memory
        return

    def train_step(self, rank, writer):
        device = torch.device("cuda:" + str(rank // self.args.n_processes_per_gpu))
        print('Running on device: ', device)
        torch.cuda.set_device(device)
        torch.set_default_tensor_type(torch.FloatTensor)
        self.setup(rank, self.args.n_processes_per_gpu * self.args.n_gpu, device)

        env = SpGcnEnv(self.args, device, writer=writer, writer_counter=self.global_writer_quality_count,
                       win_event_counter=self.global_win_event_count)
        # Create shared network

        model = GcnEdgeAC(self.cfg, self.args, device, writer=writer)
        model.cuda(device)
        shared_model = DDP(model, device_ids=[device], find_unused_parameters=True)

        if not self.args.no_fe_extr_optim:
            # optimizers
            OptimizerContainer = namedtuple('OptimizerContainer', ('actor', 'embeddings', 'critic', 'temperature'))
        else:
            OptimizerContainer = namedtuple('OptimizerContainer', ('actor', 'critic', 'temperature'))
        actor_optimizer = torch.optim.Adam(shared_model.module.actor.parameters(),
                                           lr=self.cfg.actor_lr,
                                           betas=self.cfg.actor_betas)
        if not self.args.no_fe_extr_optim:
            embeddings_optimizer = torch.optim.Adam(shared_model.module.fe_ext.parameters(),
                                               lr=self.cfg.embeddings_lr,
                                               betas=self.cfg.actor_betas)
        critic_optimizer = torch.optim.Adam(shared_model.module.critic.parameters(),
                                            lr=self.cfg.critic_lr,
                                            betas=self.cfg.critic_betas)
        temp_optimizer = torch.optim.Adam([shared_model.module.log_alpha],
                                          lr=self.cfg.alpha_lr,
                                          betas=self.cfg.alpha_betas)

        if not self.args.no_fe_extr_optim:
            optimizers = OptimizerContainer(actor_optimizer, embeddings_optimizer, critic_optimizer, temp_optimizer)
        else:
            optimizers = OptimizerContainer(actor_optimizer, critic_optimizer, temp_optimizer)

        dist.barrier()

        if self.args.model_name != "":
            shared_model.module.load_state_dict(torch.load(os.path.join(self.log_dir, self.args.model_name)))
        elif self.args.model_fe_name != "":
            shared_model.module.fe_ext.load_state_dict(torch.load(os.path.join(self.save_dir, self.args.model_fe_name)))
        elif self.args.fe_extr_warmup and rank == 0:
            print('pretrain fe extractor')
            self.pretrain_embeddings(shared_model.module.fe_ext, device, writer)

        dist.barrier()

        if self.args.no_fe_extr_optim:
            for param in model.fe_ext.parameters():
                param.requires_grad = False

        dset = SpgDset(root_dir=self.args.data_dir)
        quality = self.args.stop_qual_scaling + self.args.stop_qual_offset
        best_quality = np.inf
        last_quals = []
        while self.global_count.value() <= self.args.T_max:
            dloader = DataLoader(dset, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True, num_workers=0)
            for iteration in range(len(dset)*self.args.data_update_frequency):
                # if self.global_count.value() > self.args.T_max:
                #     a=1
                if iteration % self.args.data_update_frequency == 0:
                    self.update_env_data(env, dloader, device)
                # waff_dis = torch.softmax(env.edge_features[:, 0].squeeze() + 1e-30, dim=0)
                # waff_dis = torch.softmax(env.gt_edge_weights + 0.5, dim=0)
                # waff_dis = torch.softmax(torch.ones_like(env.b_gt_edge_weights), dim=0)
                # loss_weight = torch.softmax(env.b_gt_edge_weights + 1, dim=0)
                env.reset()
                # self.target_entropy = - float(env.gt_edge_weights.shape[0])
                self.target_entropy = -10

                if self.cfg.temperature_regulation == 'follow_quality':
                    if self.beta_rule.base < quality:
                        self.beta_rule.base = quality
                    self.beta = self.beta_rule.apply(self.global_count.value(), quality)
                else:
                    self.beta = shared_model.module.alpha.detach()

                with portalocker.Lock(os.path.join(self.log_dir, 'runtime_cfg.yaml'), 'rb+', timeout=60) as fh:
                    with open(os.path.join(self.log_dir, 'runtime_cfg.yaml')) as info:
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
                            # if 'alpha_lr' in args_dict and args_dict['alpha_lr'] != self.cfg.alpha_lr:
                            #     self.cfg.alpha_lr = args_dict['alpha_lr']
                            #     adjust_learning_rate(temp_optimizer, self.cfg.alpha_lr)
                    with open(os.path.join(self.log_dir, 'runtime_cfg.yaml'), "w") as info:
                        yaml.dump(args_dict, info)

                    # flush and sync to filesystem
                    fh.flush()
                    os.fsync(fh.fileno())

                if rank == 0 and self.args.safe_model:
                    if self.args.model_name_dest != "":
                        torch.save(shared_model.module.state_dict(),
                                   os.path.join(self.log_dir, self.args.model_name_dest))
                    else:
                        torch.save(shared_model.module.state_dict(), os.path.join(self.log_dir, 'agent_model'))

                state = env.get_state()
                while not env.done:
                    # Calculate policy and values
                    post_stats = True if (self.global_writer_count.value() + 1) % 30 == 0 and env.counter == 0 else False
                    post_model = True if (self.global_writer_count.value() + 1) % 100 == 0 and env.counter == 0 else False
                    post_stats &= self.memory.is_full()
                    post_model &= self.memory.is_full()
                    round_n = env.counter
                    # post_stats=True
                    # sample action for data collection
                    distr = None
                    if self.global_count.value() < self.cfg.num_seed_steps:
                        action = torch.rand_like(env.b_current_edge_weights)
                    else:
                        distr, _, _, action, _ = self.agent_forward(env, shared_model, state=state, grad=False,
                                                             post_input=post_stats, post_model=post_model)

                    logg_dict = {'temperature': self.beta}
                    logg_dict['alpha'] = shared_model.module.alpha.item()
                    if distr is not None:
                        logg_dict['mean_loc'] = distr.loc.mean().item()
                        logg_dict['mean_scale'] = distr.scale.mean().item()

                    if self.global_count.value() >= self.cfg.num_seed_steps and self.memory.is_full():
                        for i in range(self.args.n_updates_per_step):
                            self._step(self.memory, optimizers, env, shared_model, self.global_count.value(), writer=writer)
                            self.global_writer_loss_count.increment()

                    next_state, reward, quality = env.execute_action(action, logg_dict, post_stats=post_stats)
                    # next_state, reward, quality = env.execute_action(torch.sigmoid(distr.loc), logg_dict, post_stats=post_stats)

                    last_quals.append(quality)
                    if len(last_quals) > 10:
                        last_quals.pop(0)

                    if self.args.add_noise:
                        noise = torch.randn_like(reward) * shared_model.module.alpha.item() * 0.4
                        reward = reward + noise

                    self.memory.push(self.state_to_cpu(state), action, reward, self.state_to_cpu(next_state), env.done)
                    state = next_state

                self.global_count.increment()
                if rank == 0:
                    self.global_writer_count.increment()

        dist.barrier()
        if rank == 0:
            self.memory.clear()
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
        state = list(state)
        for i in range(len(state)):
            if torch.is_tensor(state[i]):
                state[i] = state[i].cpu()
        return state

    def state_to_cuda(self, state, device):
        state = list(state)
        for i in range(len(state)):
            if torch.is_tensor(state[i]):
                state[i] = state[i].to(device)
        return state


class SacValidate(object):

    def __init__(self, cfg, args, save_dir):
        super(SacValidate, self).__init__()
        self.cfg = cfg
        self.args = args
        self.save_dir = save_dir

    def cleanup(self):
        dist.destroy_process_group()

    def update_env_data(self, env, dloader, device):
        raw, gt, indices = next(iter(dloader))
        raw, gt = raw.to(device), gt.to(device)
        edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling = dloader.dataset.get_graphs(indices, device)
        angles = None
        env.update_data(edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, angles, gt)

    def agent_forward(self, env, model, state, actions=None, grad=True, post_input=False, post_model=False, policy_opt=False):
        with torch.set_grad_enabled(grad):
            state_pixels, edge_ids, sp_indices, edge_angles, sub_graphs, sep_subgraphs, counter, b_gt_edge_weights = self.state_to_cuda(state, env.device)
            if actions is not None:
                actions = actions.to(model.device)
            counter /= self.args.max_episode_length
            raw = state_pixels[:, 0, ...].unsqueeze(1)
            # model.module.writer.add_graph(model.module, (raw, b_gt_edge_weights, sp_indices, edge_ids, edge_angles, counter, sub_graphs, sep_subgraphs, actions, post_input, policy_opt), verbose=False)
            return model(raw,
                         edge_index=edge_ids,
                         angles=edge_angles,
                         round_n=counter,
                         actions=actions,
                         sp_indices=sp_indices,
                         gt_edges=b_gt_edge_weights,
                         sub_graphs=sub_graphs,
                         sep_subgraphs=sep_subgraphs,
                         post_input=post_input,
                         policy_opt=policy_opt and grad)

    # Acts and trains model
    def validate_seeds(self):
        rns = torch.randint(0, 2 ** 32, torch.Size([10]))
        best_qual = np.inf
        best_seed = -1
        with open(os.path.join(self.save_dir, 'result.txt'), "w") as info:
            for i, rn in enumerate(rns):
                set_seed_everywhere(rn.item())
                abs_diffs, rel_diffs, mean_size, mean_n_larger_thresh = self.validate()
                qual = sum(rel_diffs) / len(rel_diffs)
                if qual < best_qual:
                    best_qual = qual
                    best_seed = rn.item()

                res = '\nseed is: ' + str(rn.item()) + " with abs_diffs of: " \
                      + str(sum(abs_diffs)) + ' :: and rel_diffs of: ' + str(qual) + " Num of diffs larger than 0.5: " \
                      + str(mean_n_larger_thresh) + "/" + str(mean_size)
                print(res)
                info.write(res)

        return best_seed, best_qual

    def validate(self):
        self.device = torch.device("cuda:0")
        model = GcnEdgeAC(self.cfg, self.args, self.device)
        thresh = 0.5

        assert self.args.model_name != ""
        model.load_state_dict(torch.load(os.path.join(self.save_dir, self.args.model_name)))

        model.cuda(self.device)
        for param in model.parameters():
            param.requires_grad = False
        dloader = DataLoader(SpgDset(root_dir=self.args.data_dir), batch_size=1, shuffle=True, pin_memory=True,
                             num_workers=0)
        env = SpGcnEnv(self.args, self.device)
        abs_diffs, rel_diffs, sizes, n_larger_thresh = [], [], [], []

        for i in range(len(dloader)):
            self.update_env_data(env, dloader, self.device)
            env.reset()
            state = env.get_state()

            distr, _, _, _, _ = self.agent_forward(env, model, state=state, grad=False)
            actions = torch.sigmoid(distr.loc)

            diff = (actions - env.b_gt_edge_weights).squeeze().abs()

            abs_diffs.append(diff.sum().item())
            rel_diffs.append(diff.mean().item())
            sizes.append(len(diff))
            n_larger_thresh.append((diff>thresh).float().sum().item())

        mean_size = sum(sizes) / len(sizes)
        mean_n_larger_thresh = sum(n_larger_thresh) / len(n_larger_thresh)
        return abs_diffs, rel_diffs, mean_size, mean_n_larger_thresh

    def state_to_cpu(self, state):
        state = list(state)
        for i in range(len(state)):
            if torch.is_tensor(state[i]):
                state[i] = state[i].cpu()
        return state

    def state_to_cuda(self, state):
        state = list(state)
        for i in range(len(state)):
            if torch.is_tensor(state[i]):
                state[i] = state[i].to(self.device)
        return state