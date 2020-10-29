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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses import ContrastiveLoss
from losses.rag_contrastive_triplet_loss import ContrastiveTripletLoss


class AgentSacTrainer_test_sg_global(object):

    def __init__(self, cfg, global_count, global_writer_loss_count,
                 global_writer_quality_count, action_stats_count, global_writer_count, save_dir):
        super(AgentSacTrainer_test_sg_global, self).__init__()

        self.cfg = cfg
        self.global_count = global_count
        self.global_writer_loss_count = global_writer_loss_count
        self.global_writer_quality_count = global_writer_quality_count
        self.action_stats_count = action_stats_count
        self.global_writer_count = global_writer_count
        self.contr_trpl_loss = ContrastiveTripletLoss(delta_var=0.5)
        self.contr_loss = ContrastiveLoss(delta_var=0.5, delta_dist=1.5)

        self.memory = TransitionData_ts(capacity=self.cfg.trainer.t_max)
        # self.eps = self.args.init_epsilon
        self.save_dir = save_dir

    def setup(self, rank, world_size):
        # BLAS setup
        os.environ['OMP_NUM_THREADS'] = '10'
        os.environ['MKL_NUM_THREADS'] = '10'

        # assert torch.cuda.device_count() == 1
        torch.set_default_tensor_type('torch.FloatTensor')

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = self.cfg.gen.master_port

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def pretrain_embeddings_gt(self, model, device, writer=None):
        dset = SpgDset(root_dir=self.cfg.gen.data_dir)
        dloader = DataLoader(dset, batch_size=self.cfg.fe.warmup.batch_size, shuffle=True, pin_memory=True,
                             num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        max_p = torch.nn.MaxPool2d(3, padding=1, stride=1)
        sheduler = ReduceLROnPlateau(optimizer)
        acc_loss = 0
        iteration = 0

        while iteration <= self.cfg.fe.warmup.n_iterations:
            for it, (raw, gt, sp_seg, indices) in enumerate(dloader):
                raw, gt, sp_seg = raw.to(device), gt.to(device), sp_seg.to(device)
                sp_seg_edge = torch.cat([(-max_p(-sp_seg) != sp_seg).float(), (max_p(sp_seg) != sp_seg).float()], 1)
                embeddings = model(torch.cat([raw, sp_seg_edge], 1))
                loss = self.contr_loss(embeddings, gt.long().squeeze(1))

                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()
                acc_loss += loss.item()

                if writer is not None:
                    writer.add_scalar("fe_warm_start/loss", loss.item(), iteration)
                    writer.add_scalar("fe_warm_start/lr", optimizer.param_groups[0]['lr'], iteration)
                    if it % 50 == 0:
                        plt.clf()
                        fig = plt.figure(frameon=False)
                        plt.imshow(sp_seg[0].detach().squeeze().cpu().numpy())
                        plt.colorbar()
                        writer.add_figure("image/sp_seg", fig, iteration // 50)
                if it % 10 == 0:
                    sheduler.step(acc_loss / 10)
                    acc_loss = 0
                iteration += 1
                if iteration > self.cfg.fe.warmup.n_iterations:
                    break

                del loss
                del embeddings
        return

    def pretrain_embeddings_sp(self, model, device, writer=None):
        dset = SpgDset(self.args.data_dir, self.cfg.fe.patch_manager, self.cfg.fe.patch_stride, self.cfg.fe.patch_shape)
        dloader = DataLoader(dset, batch_size=self.cfg.fe.warmup.batch_size, shuffle=True, pin_memory=True,
                             num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        max_p = torch.nn.MaxPool2d(3, padding=1, stride=1)
        sheduler = ReduceLROnPlateau(optimizer)
        acc_loss = 0

        for i in range(self.cfg.fe.warmup.n_iterations):
            print(f"fe ext wu iter: {i}")
            for it, (raw, gt, sp_seg, indices) in enumerate(dloader):
                raw, gt, sp_seg = raw.to(device), gt.to(device), sp_seg.to(device)
                sp_seg_edge = torch.cat([(-max_p(-sp_seg) != sp_seg).float(), (max_p(sp_seg) != sp_seg).float()], 1)
                embeddings = model(torch.cat([raw, sp_seg_edge], 1), True if it % 500 == 0 else False)

                loss = self.contr_loss(embeddings, sp_seg.long().squeeze(1))

                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()
                acc_loss += loss.item()

                if writer is not None:
                    writer.add_scalar("fe_warm_start/loss", loss.item(), (len(dloader) * i) + it)
                    writer.add_scalar("fe_warm_start/lr", optimizer.param_groups[0]['lr'], (len(dloader) * i) + it)
                    if it % 500 == 0:
                        plt.clf()
                        fig = plt.figure(frameon=False)
                        plt.imshow(sp_seg[0].detach().squeeze().cpu().numpy())
                        plt.colorbar()
                        writer.add_figure("image/sp_seg", fig, ((len(dloader) * i) + it) // 500)

                if it % 10 == 0:
                    sheduler.step(acc_loss / 10)
                    acc_loss = 0

    def cleanup(self):
        dist.destroy_process_group()

    def update_env_data(self, env, dloader, device):
        raw, gt, sp_seg, indices = next(iter(dloader))
        raw, gt, sp_seg = raw.to(device), gt.to(device), sp_seg.to(device)
        edges, edge_feat, diff_to_gt, gt_edge_weights = dloader.dataset.get_graphs(indices, device)
        env.update_data(edges, edge_feat, diff_to_gt, gt_edge_weights, sp_seg, raw, gt)

    def agent_forward(self, env, model, state, actions=None, grad=True, post_input=False, post_model=False,
                      policy_opt=False, embeddings_opt=False):
        with torch.set_grad_enabled(grad):
            raw, sp_seg, edge_ids, sp_indices, edge_angles, sub_graphs, sep_subgraphs, counter, b_gt_edge_weights, edge_offsets = self.state_to_cuda(
                state, env.device)
            if actions is not None:
                actions = actions.to(model.module.device)
            counter /= self.cfg.trainer.max_episode_length
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
            edges = obs[2][:, obs[-1][i]:obs[-1][i + 1]]
            edges = edges - edges.min()

            attr_weight_del = weights[obs[-1][i]:obs[-1][i + 1]] < self.cfg.sac.weight_tolerance_attr
            attr_weights = weights[obs[-1][i]:obs[-1][i + 1]][attr_weight_del]
            if len(attr_weights) != 0:
                attr_weights = attr_weights - attr_weights.min()
                max = attr_weights.max()
                max = max if max != 0 else 1e-16
                attr_weights = attr_weights / max
                attr_weights += 1e-16  # make sure all edges exist in graph
                direct_attr = edges[:, attr_weight_del].numpy()
                b_attr_edges.append(self._get_connected_paths(direct_attr, attr_weights, edges.max() + 1))
            else:
                b_attr_edges.append(None)

            rep_weight_del = weights[obs[-1][i]:obs[-1][i + 1]] <= self.cfg.sac.weight_tolerance_rep
            rep_weights = weights[obs[-1][i]:obs[-1][i + 1]][rep_weight_del]
            if len(rep_weights) != 0:
                rep_weights = rep_weights - rep_weights.min()
                max = rep_weights.max()
                max = max if max != 0 else 1e-16
                rep_weights = rep_weights / max
                rep_weights += 1e-16  # make sure all edges exist in graph
                direct_rep = edges[:, rep_weight_del].numpy()
                b_rep_edges.append(
                    self._get_connected_paths(direct_rep, rep_weights, edges.max() + 1, get_repulsive=True))
            else:
                b_rep_edges.append(None)

        return self.contr_trpl_loss(
            embeddings, obs[1].long().to(embeddings.device), (b_attr_edges, b_rep_edges))

    def get_embed_loss_contr(self, weights, env, embeddings):
        segs = env.get_current_soln(weights)
        return self.contr_loss(embeddings, segs.long().to(embeddings.device))

    def update_embeddings(self, obs, env, model, optimizers):
        distribution, actor_Q1, actor_Q2, action, embeddings, side_loss = \
            self.agent_forward(env, model, grad=False, state=obs, policy_opt=False, embeddings_opt=True)
        weights = distribution.loc.detach()
        loss = self.get_embed_loss_contr(weights, env, embeddings)

        optimizers.embeddings.zero_grad()
        loss.backward()
        optimizers.embeddings.step()
        return loss.item()

    def update_critic(self, obs, action, reward, next_obs, not_done, env, model, optimizers):
        distribution, target_Q1, target_Q2, next_action, _, side_loss = self.agent_forward(env, model, state=next_obs)
        current_Q1, current_Q2, side_loss = self.agent_forward(env, model, state=obs, actions=action)

        log_prob = distribution.log_prob(next_action)
        critic_loss = torch.tensor([0.0], device=target_Q1[0].device)
        mean_reward = 0

        for i, sz in enumerate(self.cfg.sac.s_subgraph):
            _log_prob = log_prob[next_obs[5][i].view(-1, sz)].sum(-1)

            target_V = torch.min(target_Q1[i], target_Q2[i]) - model.module.alpha[i].detach() * _log_prob

            target_Q = reward[i] + (not_done * self.cfg.sac.discount * target_V)
            target_Q = target_Q.detach()

            critic_loss = critic_loss + (F.mse_loss(current_Q1[i], target_Q) + F.mse_loss(current_Q2[i], target_Q))# / 2) + self.cfg.sac.sl_beta * side_loss
            mean_reward += reward[i].mean()
        # critic_loss = critic_loss / len(self.cfg.sac.s_subgraph)
        optimizers.critic.zero_grad()
        critic_loss.backward()
        optimizers.critic.step()

        return critic_loss.item(), mean_reward / len(self.cfg.sac.s_subgraph)

    def update_actor_and_alpha(self, obs, env, model, optimizers, embeddings_opt=False):
        distribution, actor_Q1, actor_Q2, action, side_loss = \
            self.agent_forward(env, model, state=obs, policy_opt=True, embeddings_opt=embeddings_opt)

        log_prob = distribution.log_prob(action)
        actor_loss = torch.tensor([0.0], device=actor_Q1[0].device)
        alpha_loss = torch.tensor([0.0], device=actor_Q1[0].device)

        for i, sz in enumerate(self.cfg.sac.s_subgraph):
            actor_Q = torch.min(actor_Q1[i], actor_Q2[i])

            _log_prob = log_prob[obs[5][i].view(-1, sz)].sum(-1)
            loss = (model.module.alpha[i].detach() * _log_prob - actor_Q[i]).mean()

            actor_loss = actor_loss + loss# + self.cfg.sac.sl_beta * side_loss

        actor_loss = actor_loss / len(self.cfg.sac.s_subgraph)
        alpha_loss = alpha_loss / len(self.cfg.sac.s_subgraph)

        optimizers.actor.zero_grad()
        actor_loss.backward()
        optimizers.actor.step()

        for i, sz in enumerate(self.cfg.sac.s_subgraph):
            _log_prob = log_prob[obs[5][i].view(-1, sz)].sum(-1)
            alpha_loss = alpha_loss + (model.module.alpha[i] *
                                       (-_log_prob + self.cfg.sac.s_subgraph[i]).detach()).mean()

        optimizers.temperature.zero_grad()
        alpha_loss.backward()
        optimizers.temperature.step()

        return actor_loss.item(), alpha_loss.item()

    def _step(self, replay_buffer, optimizers, mov_sum_loss, env, model, step, writer=None):

        (obs, action, reward, next_obs, done), sample_idx = replay_buffer.sample()
        not_done = int(not done)
        n_prep_steps = self.cfg.trainer.t_max - self.cfg.fe.update_after_steps
        embeddings_opt = step - n_prep_steps > 0 and (step - n_prep_steps) % self.cfg.fe.update_frequency == 0

        if "extra" in self.cfg.fe.optim:
            if embeddings_opt:
                embedd_loss = self.update_embeddings(obs, env, model, optimizers)
                mov_sum_loss.embeddings.apply(embedd_loss)
                optimizers.embed_shed.step(mov_sum_loss.embeddings.avg)
                if writer is not None:
                    writer.add_scalar("loss/embedd", embedd_loss, self.global_writer_loss_count.value())
                return

        critic_loss, mean_reward = self.update_critic(obs, action, reward, next_obs, not_done, env, model, optimizers)
        mov_sum_loss.critic.apply(critic_loss)
        optimizers.critic_shed.step(mov_sum_loss.critic.avg)
        replay_buffer.report_sample_loss(critic_loss + mean_reward, sample_idx)

        if step % self.cfg.sac.actor_update_frequency == 0:
            actor_loss, alpha_loss = self.update_actor_and_alpha(obs, env, model, optimizers, embeddings_opt)
            mov_sum_loss.actor.apply(actor_loss)
            mov_sum_loss.temperature.apply(alpha_loss)
            optimizers.temp_shed.step(mov_sum_loss.actor.avg)
            optimizers.temp_shed.step(mov_sum_loss.temperature.avg)
            if writer is not None:
                writer.add_scalar("loss/actor", actor_loss, self.global_writer_loss_count.value())
                writer.add_scalar("loss/temperature", alpha_loss, self.global_writer_loss_count.value())

        if step % self.cfg.sac.critic_target_update_frequency == 0:
            soft_update_params(model.module.critic, model.module.critic_tgt, self.cfg.sac.critic_tau)

        if writer is not None:
            writer.add_scalar("loss/critic", critic_loss, self.global_writer_loss_count.value())

    # Acts and trains model
    def train(self, rank, start_time, return_dict, rn):

        self.log_dir = os.path.join(self.save_dir, 'logs', '_' + str(rn))
        writer = None
        if rank == 0:
            writer = SummaryWriter(logdir=self.log_dir)
            writer.add_text("config", self.cfg.pretty(), 0)
            copyfile(os.path.join(self.save_dir, 'runtime_cfg.yaml'),
                     os.path.join(self.log_dir, 'runtime_cfg.yaml'))

            self.global_count.reset()
            self.global_writer_loss_count.reset()
            self.global_writer_quality_count.reset()
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
        device = torch.device("cuda:" + str(rank // self.cfg.gen.n_processes_per_gpu))
        print('Running on device: ', device)
        torch.cuda.set_device(device)
        torch.set_default_tensor_type(torch.FloatTensor)
        self.setup(rank, self.cfg.gen.n_processes_per_gpu * self.cfg.gen.n_gpu)

        env = SpGcnEnv(self.cfg, device, writer=writer, writer_counter=self.global_writer_quality_count)
        # Create shared network

        model = GcnEdgeAC(self.cfg, device, writer=writer)
        model.cuda(device)
        shared_model = DDP(model, device_ids=[device], find_unused_parameters=True)
        if 'extra' in self.cfg.fe.optim:
            # optimizers
            MovSumLosses = namedtuple('mov_avg_losses', ('actor', 'embeddings', 'critic', 'temperature'))
            OptimizerContainer = namedtuple('OptimizerContainer', ('actor', 'embeddings', 'critic', 'temperature',
                                                                   'actor_shed', 'embed_shed', 'critic_shed',
                                                                   'temp_shed'))
        else:
            MovSumLosses = namedtuple('mov_avg_losses', ('actor', 'critic', 'temperature'))
            OptimizerContainer = namedtuple('OptimizerContainer', ('actor', 'critic', 'temperature',
                                                                   'actor_shed', 'critic_shed', 'temp_shed'))
        if "rl_loss" == self.cfg.fe.optim:
            actor_optimizer = torch.optim.Adam(list(shared_model.module.actor.parameters())
                                               + list(shared_model.module.fe_ext.parameters()),
                                               lr=self.cfg.sac.actor_lr,
                                               betas=self.cfg.sac.actor_betas)
        else:
            actor_optimizer = torch.optim.Adam(shared_model.module.actor.parameters(),
                                               lr=self.cfg.sac.actor_lr,
                                               betas=self.cfg.sac.actor_betas)
        if "extra" in self.cfg.fe.optim:
            embeddings_optimizer = torch.optim.Adam(shared_model.module.fe_ext.parameters(),
                                                    lr=self.cfg.fe.lr,
                                                    betas=self.cfg.fe.betas)
        critic_optimizer = torch.optim.Adam(shared_model.module.critic.parameters(),
                                            lr=self.cfg.sac.critic_lr,
                                            betas=self.cfg.sac.critic_betas)
        temp_optimizer = torch.optim.Adam([shared_model.module.log_alpha],
                                          lr=self.cfg.sac.alpha_lr,
                                          betas=self.cfg.sac.alpha_betas)

        if "extra" in self.cfg.fe.optim:
            mov_sum_losses = MovSumLosses(RunningAverage(), RunningAverage(), RunningAverage(), RunningAverage())
            optimizers = OptimizerContainer(actor_optimizer, embeddings_optimizer, critic_optimizer, temp_optimizer,
                                            ReduceLROnPlateau(actor_optimizer), ReduceLROnPlateau(embeddings_optimizer),
                                            ReduceLROnPlateau(critic_optimizer), ReduceLROnPlateau(temp_optimizer))
        else:
            mov_sum_losses = MovSumLosses(RunningAverage(), RunningAverage(), RunningAverage())
            optimizers = OptimizerContainer(actor_optimizer, critic_optimizer, temp_optimizer,
                                            ReduceLROnPlateau(actor_optimizer),
                                            ReduceLROnPlateau(critic_optimizer), ReduceLROnPlateau(temp_optimizer))

        dist.barrier()

        if self.cfg.gen.resume:
            shared_model.module.load_state_dict(torch.load(os.path.join(self.log_dir, self.cfg.gen.model_name)))
        elif self.cfg.fe.load_pretrained:
            shared_model.module.fe_ext.load_state_dict(torch.load(os.path.join(self.save_dir, self.cfg.fe.model_name)))
        elif 'warmup' in self.cfg.fe and rank == 0:
            print('pretrain fe extractor')
            self.pretrain_embeddings_gt(shared_model.module.fe_ext, device, writer)
            torch.save(shared_model.module.fe_ext.state_dict(),
                       os.path.join(self.save_dir, self.cfg.fe.model_name))
        dist.barrier()

        if "none" == self.cfg.fe.optim:
            for param in shared_model.module.fe_ext.parameters():
                param.requires_grad = False

        dset = SpgDset(self.cfg.gen.data_dir)
        step = 0
        while self.global_count.value() <= self.cfg.trainer.T_max:
            dloader = DataLoader(dset, batch_size=self.cfg.trainer.batch_size, shuffle=True, pin_memory=True,
                                 num_workers=0)
            for iteration in range(len(dset) * self.cfg.trainer.data_update_frequency):
                # if self.global_count.value() > self.args.T_max:
                #     a=1
                if iteration % self.cfg.trainer.data_update_frequency == 0:
                    self.update_env_data(env, dloader, device)
                # waff_dis = torch.softmax(env.edge_features[:, 0].squeeze() + 1e-30, dim=0)
                # waff_dis = torch.softmax(env.gt_edge_weights + 0.5, dim=0)
                # waff_dis = torch.softmax(torch.ones_like(env.b_gt_edge_weights), dim=0)
                # loss_weight = torch.softmax(env.b_gt_edge_weights + 1, dim=0)
                env.reset()
                self.update_rt_vars(critic_optimizer, actor_optimizer)
                if rank == 0 and self.cfg.rt_vars.safe_model:
                    if self.cfg.gen.model_name != "":
                        torch.save(shared_model.module.state_dict(),
                                   os.path.join(self.log_dir, self.cfg.gen.model_name))
                    else:
                        torch.save(shared_model.module.state_dict(), os.path.join(self.log_dir, 'agent_model'))

                state = env.get_state()
                while not env.done:
                    # Calculate policy and values
                    post_stats = True if (self.global_writer_count.value() + 1) % self.cfg.trainer.post_stats_frequency == 0 \
                        else False
                    post_model = True if (self.global_writer_count.value() + 1) % self.cfg.trainer.post_model_frequency == 0 \
                        else False
                    post_stats &= self.memory.is_full()
                    post_model &= self.memory.is_full()
                    distr = None
                    if not self.memory.is_full():
                        action = torch.rand_like(env.current_edge_weights)
                    else:
                        distr, _, _, action, _, _ = self.agent_forward(env, shared_model, state=state, grad=False,
                                                                    post_input=post_stats, post_model=post_model)

                    logg_dict = {}
                    if post_stats:
                        for i in range(len(self.cfg.sac.s_subgraph)):
                            logg_dict['alpha_' + str(i)] = shared_model.module.alpha[i].item()
                        if distr is not None:
                            logg_dict['mean_loc'] = distr.loc.mean().item()
                            logg_dict['mean_scale'] = distr.scale.mean().item()

                    if self.memory.is_full():
                        for i in range(self.cfg.trainer.n_updates_per_step):
                            self._step(self.memory, optimizers, mov_sum_losses, env, shared_model,
                                       step, writer=writer)
                            self.global_writer_loss_count.increment()

                    next_state, reward = env.execute_action(action, logg_dict, post_stats=post_stats)
                    # next_state, reward, quality = env.execute_action(torch.sigmoid(distr.loc), logg_dict, post_stats=post_stats)

                    if self.cfg.rt_vars.add_noise:
                        noise = torch.randn_like(reward) * 0.2
                        reward = reward + noise

                    self.memory.push(self.state_to_cpu(state), action, reward, self.state_to_cpu(next_state), env.done)
                    state = next_state

                self.global_count.increment()
                step += 1
                if rank == 0:
                    self.global_writer_count.increment()
                if step > self.cfg.trainer.T_max:
                    break

        dist.barrier()
        if rank == 0:
            self.memory.clear()
            if not self.cfg.gen.cross_validate_hp and not self.cfg.gen.test_score_only and not self.cfg.gen.no_save:
                # pass
                if self.cfg.gen.model_name != "":
                    torch.save(shared_model.state_dict(), os.path.join(self.log_dir, self.cfg.gen.model_name))
                    print('saved')
                else:
                    torch.save(shared_model.state_dict(), os.path.join(self.log_dir, 'agent_model'))

        self.cleanup()
        return sum(env.acc_reward) / len(env.acc_reward)

    def state_to_cpu(self, state):
        state = list(state)
        for i in range(len(state)):
            if torch.is_tensor(state[i]):
                state[i] = state[i].cpu()
            elif isinstance(state[i], list) or isinstance(state[i], tuple):
                state[i] = self.state_to_cpu(state[i])
        return state

    def state_to_cuda(self, state, device):
        state = list(state)
        for i in range(len(state)):
            if torch.is_tensor(state[i]):
                state[i] = state[i].to(device)
            elif isinstance(state[i], list) or isinstance(state[i], tuple):
                state[i] = self.state_to_cuda(state[i], device)
        return state

    def update_rt_vars(self, critic_optimizer, actor_optimizer):
        with portalocker.Lock(os.path.join(self.log_dir, 'runtime_cfg.yaml'), 'rb+', timeout=60) as fh:
            with open(os.path.join(self.log_dir, 'runtime_cfg.yaml')) as info:
                args_dict = yaml.full_load(info)
                if args_dict is not None:
                    if 'safe_model' in args_dict:
                        self.cfg.rt_vars.safe_model = args_dict['safe_model']
                        args_dict['safe_model'] = False
                    if 'add_noise' in args_dict:
                        self.cfg.rt_vars.add_noise = args_dict['add_noise']
                    if 'critic_lr' in args_dict and args_dict['critic_lr'] != self.cfg.sac.critic_lr:
                        self.cfg.sac.critic_lr = args_dict['critic_lr']
                        adjust_learning_rate(critic_optimizer, self.cfg.sac.critic_lr)
                    if 'actor_lr' in args_dict and args_dict['actor_lr'] != self.cfg.sac.actor_lr:
                        self.cfg.sac.actor_lr = args_dict['actor_lr']
                        adjust_learning_rate(actor_optimizer, self.cfg.sac.actor_lr)
            with open(os.path.join(self.log_dir, 'runtime_cfg.yaml'), "w") as info:
                yaml.dump(args_dict, info)

            # flush and sync to filesystem
            fh.flush()
            os.fsync(fh.fileno())


class SacValidate(object):

    def __init__(self, cfg, save_dir):
        super(SacValidate, self).__init__()
        self.cfg = cfg
        self.save_dir = save_dir

    def cleanup(self):
        dist.destroy_process_group()

    def update_env_data(self, env, dloader, device):
        raw, gt, indices = next(iter(dloader))
        raw, gt = raw.to(device), gt.to(device)
        edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling = dloader.dataset.get_graphs(indices, device)
        angles = None
        env.update_data(edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, angles, gt)

    def agent_forward(self, env, model, state, actions=None, grad=True, post_input=False, post_model=False,
                      policy_opt=False):
        with torch.set_grad_enabled(grad):
            state_pixels, edge_ids, sp_indices, edge_angles, sub_graphs, sep_subgraphs, counter, b_gt_edge_weights = self.state_to_cuda(
                state, env.device)
            if actions is not None:
                actions = actions.to(model.device)
            counter /= self.cfg.trainer.max_episode_length
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
        model = GcnEdgeAC(self.cfg, self.device)
        thresh = 0.5

        assert self.cfg.gen.model_name != ""
        model.load_state_dict(torch.load(os.path.join(self.save_dir, self.cfg.gen.model_name)))

        model.cuda(self.device)
        for param in model.parameters():
            param.requires_grad = False
        dloader = DataLoader(SpgDset(root_dir=self.cfg.gen.data_dir), batch_size=1, shuffle=True, pin_memory=True,
                             num_workers=0)
        env = SpGcnEnv(self.cfg, self.device)
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
            n_larger_thresh.append((diff > thresh).float().sum().item())

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
