from tqdm import tqdm
import numpy as np
from agents.exploration_functions import NaiveDecay, ActionPathTreeNodes, ExpSawtoothEpsDecay
import os
import shutil

import torch
from utils.general import Counter
from models.GCNNs.mc_glbl_edge_costs import GcnEdgeAngle1dPQV
from models.GCNNs.mc_glbl_edge_costs import WrappedGcnEdgeAngle1dPQV
from models.GCNNs.dueling_networks import WrappedGcnEdgeAngle1dPQA_dueling
from models.GCNNs.dueling_networks_1 import WrappedGcnEdgeAngle1dPQA_dueling_1
from optimizers.shared_rmsprob import SharedRMSprop
from torch import multiprocessing as mp
from agents.acer import AgentAcerTrainer
from agents.acer_continuous import AgentAcerContinuousTrainer
from mu_net.criteria.contrastive_loss import ContrastiveLoss
from torch.utils.data import DataLoader
import yaml
from data.disjoint_discs import MultiDiscSpGraphDset

class TrainACER(object):

    def __init__(self, args, eps=0.9):
        super(TrainACER, self).__init__()
        self.args = args
        self.eps = eps
        self.device = torch.device("cuda:0")

    def train(self, time):
        # Creating directories.
        save_dir = os.path.join(self.args.base_dir, 'results/acer', self.args.target_dir)
        log_dir = os.path.join(save_dir, 'logs')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if os.path.exists(os.path.join(save_dir, 'config.yaml')):
            os.remove(os.path.join(save_dir, 'config.yaml'))
        print(' ' * 26 + 'Options')

        # Saving parameters
        with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
            for k, v in vars(self.args).items():
                print(' ' * 26 + k + ': ' + str(v))
                f.write(k + ' : ' + str(v) + '\n')

        with open(os.path.join(save_dir, 'config.yaml'), "w") as info:
            documents = yaml.dump(vars(self.args), info)

        torch.manual_seed(self.args.seed)
        global_count = Counter()  # Global shared counter
        global_writer_loss_count = Counter()  # Global shared counter
        global_writer_quality_count = Counter()  # Global shared counter
        global_win_event_count = Counter()  # Global shared counter

        # Create average network
        if self.args.algorithm == "acer":
            shared_average_model = WrappedGcnEdgeAngle1dPQV(self.args.n_raw_channels, self.args.n_embedding_features,
                                                            self.args.n_edge_features, self.args.n_actions,
                                                            self.device, None)
        else:
            shared_average_model = WrappedGcnEdgeAngle1dPQA_dueling_1(self.args.n_raw_channels,
                                                                    self.args.n_embedding_features,
                                                                    self.args.n_edge_features, 1, self.args.exp_steps,
                                                                    self.args.p_sigma, self.device,
                                                                    self.args.density_eval_range, None)
        for param in shared_average_model.parameters():
            param.requires_grad = False

        shared_average_model.share_memory()

        # Create optimiser for shared network parameters with shared statistics
        # optimizer = SharedAdam(shared_model.parameters(), lr=self.args.Adam_lr, betas=self.args.Adam_betas,
        #                        weight_decay=self.args.Adam_weight_decay)
        shared_average_model.cuda(device=self.device)

        # Start validation agent
        processes = []

        # ctx = mp.get_context('spawn')
        # q = ctx.Queue()
        # e = ctx.Event()
        # p = ctx.Process(target=send_and_delete_tensors, args=(q, e, torch.cuda.IntTensor, 5))
        # p.start()
        if not self.args.evaluate:
            # Start training agents
            manager = mp.Manager()
            return_dict = manager.dict()
            if self.args.algorithm == "acer":
                trainer = AgentAcerTrainer(self.args, shared_average_model, global_count, global_writer_loss_count,
                                           global_writer_quality_count, global_win_event_count=global_win_event_count,
                                           save_dir=save_dir)
            else:
                trainer = AgentAcerContinuousTrainer(self.args, shared_average_model, global_count,
                                                     global_writer_loss_count,
                                                     global_writer_quality_count,
                                                     global_win_event_count=global_win_event_count,
                                                     save_dir=save_dir)
            for rank in range(0, self.args.num_processes):
                p = mp.Process(target=trainer.train, args=(rank, time, return_dict))
                p.start()
                processes.append(p)

        # Clean up
        for p in processes:
            p.join()
        if self.args.cross_validate_hp or self.args.test_score_only:
            print('Score is: ', return_dict['test_score'])
            print('Wins out of 20 trials')
            return return_dict['test_score']

