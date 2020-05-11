from tqdm import tqdm
import numpy as np
from agents.exploration_functions import NaiveDecay, ActionPathTreeNodes, ExpSawtoothEpsDecay
import os
import shutil

import torch
from utils.general import Counter
from models.GCNNs.mc_glbl_edge_costs import GcnEdgeAngle1dPQV
from models.GCNNs.mc_glbl_edge_costs import WrappedGcnEdgeAngle1dPQV
from optimizers.shared_rmsprob import SharedRMSprop
from torch import multiprocessing as mp
from agents.offpac_2_models import AgentOffpac2M
from mu_net.criteria.contrastive_loss import ContrastiveLoss
from torch.utils.data import DataLoader
from data.disjoint_discs import MultiDiscSpGraphDset

class TrainOffpac2M(object):

    def __init__(self, args, eps=0.9):
        super(TrainOffpac2M, self).__init__()
        self.args = args
        self.eps = eps
        self.device = torch.device("cuda:0")

    def train(self, time):
        # Creating directories.
        save_dir = os.path.join(self.args.base_dir, 'results/offpac2M', self.args.target_dir)
        log_dir = os.path.join(save_dir, 'logs')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        print(' ' * 26 + 'Options')

        # Saving parameters
        with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
            for k, v in vars(self.args).items():
                print(' ' * 26 + k + ': ' + str(v))
                f.write(k + ' : ' + str(v) + '\n')

        torch.manual_seed(self.args.seed)
        global_count = Counter()  # Global shared counter
        global_writer_loss_count = Counter()  # Global shared counter
        global_writer_quality_count = Counter()  # Global shared counter
        global_win_event_count = Counter()  # Global shared counter

        # Create average network
        # shared_damped_model = WrappedGcnEdgeAngle1dPQV(self.args.n_raw_channels, self.args.n_embedding_features,
        #                                                self.args.n_edge_features, self.args.n_actions,
        #                                                self.device)
        # shared_damped_model.share_memory()
        #
        # for param in shared_damped_model.parameters():
        #     param.requires_grad = False
        # # Create optimiser for shared network parameters with shared statistics
        # # optimizer = SharedAdam(shared_model.parameters(), lr=self.args.Adam_lr, betas=self.args.Adam_betas,
        # #                        weight_decay=self.args.Adam_weight_decay)
        # shared_damped_model.cuda(device=self.device)

        processes = []
        if not self.args.evaluate:
            # Start training agents
            manager = mp.Manager()
            return_dict = manager.dict()
            trainer = AgentOffpac2M(self.args, global_count, global_writer_loss_count,
                                  global_writer_quality_count, global_win_event_count, save_dir)
            for rank in range(0, self.args.num_processes):
                p = mp.Process(target=trainer.train, args=(rank, time, return_dict))
                p.start()
                processes.append(p)

        # Clean up
        for p in processes:
            p.join()
        if self.args.cross_validate_hp:
            return return_dict['test_score']
