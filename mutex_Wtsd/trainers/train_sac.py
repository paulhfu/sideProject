import os
import torch
from utils.general import Counter
from torch import multiprocessing as mp
from agents.sac import AgentSacTrainer
from agents.sac_sparse_rewards import AgentSacSRTrainer
from agents.batched_sac_sg_rew import BatchedSgRewAgentSacTrainer
from agents.batched_sac import BatchedAgentSacTrainer
from agents.sac_seed_test import AgentSacTrainer_test
from agents.sac_seed_test_sg import AgentSacTrainer_test_sg
from agents.sac_seed_test_global import AgentSacTrainer_test_global
from agents.sac_seed_test_sg_global import AgentSacTrainer_test_sg_global, SacValidate
import yaml

class TrainSAC(object):

    def __init__(self, cfg, args, eps=0.9):
        super(TrainSAC, self).__init__()
        self.cfg = cfg
        self.args = args
        self.eps = eps
        self.device = torch.device("cuda:0")

    def train(self, time):
        # Creating directories.
        save_dir = os.path.join(self.args.base_dir, 'results/sac', self.args.target_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not self.args.evaluate:
            log_dir = os.path.join(save_dir, 'logs')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            if os.path.exists(os.path.join(save_dir, 'runtime_cfg.yaml')):
                os.remove(os.path.join(save_dir, 'runtime_cfg.yaml'))

        print(self.args.pretty())
        print(self.cfg.pretty())

        with open(os.path.join(save_dir, 'config.txt'), "w") as info:
            info.write(self.args.pretty())

        if not self.args.evaluate:
            with open(os.path.join(save_dir, 'runtime_cfg.yaml'), "w") as info:
                yaml.dump(dict(self.args.runtime_config), info)

            global_count = Counter()  # Global shared counter
            global_writer_loss_count = Counter()  # Global shared counter
            global_writer_quality_count = Counter()  # Global shared counter
            global_win_event_count = Counter()  # Global shared counter
            action_stats_count = Counter()

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
            if 'seed_test_sg_global' in self.args.algorithm:
                trainer = AgentSacTrainer_test_sg_global(self.cfg, self.args, global_count,
                                                         global_writer_loss_count,
                                                         global_writer_quality_count,
                                                         global_win_event_count=global_win_event_count,
                                                         action_stats_count=action_stats_count,
                                                         save_dir=save_dir)
            elif 'seed_test_sg' in self.args.algorithm:
                trainer = AgentSacTrainer_test_sg(self.cfg, self.args, global_count, global_writer_loss_count,
                                                  global_writer_quality_count,
                                                  global_win_event_count=global_win_event_count,
                                                  action_stats_count=action_stats_count,
                                                  save_dir=save_dir)
            elif 'seed_test_global' in self.args.algorithm:
                trainer = AgentSacTrainer_test_global(self.cfg, self.args, global_count, global_writer_loss_count,
                                                      global_writer_quality_count,
                                                      global_win_event_count=global_win_event_count,
                                                      action_stats_count=action_stats_count,
                                                      save_dir=save_dir)
            elif 'seed_test' in self.args.algorithm:
                trainer = AgentSacTrainer_test(self.cfg, self.args, global_count, global_writer_loss_count,
                                               global_writer_quality_count,
                                               global_win_event_count=global_win_event_count,
                                               action_stats_count=action_stats_count,
                                               save_dir=save_dir)
            elif 'global_sparse' in self.args.algorithm:
                trainer = AgentSacSRTrainer(self.cfg, self.args, global_count, global_writer_loss_count,
                                            global_writer_quality_count, global_win_event_count=global_win_event_count,
                                            action_stats_count=action_stats_count,
                                            save_dir=save_dir)
            elif 'batched_sg_rew' in self.args.algorithm:
                trainer = BatchedSgRewAgentSacTrainer(self.cfg, self.args, global_count, global_writer_loss_count,
                                                      global_writer_quality_count,
                                                      global_win_event_count=global_win_event_count,
                                                      action_stats_count=action_stats_count,
                                                      save_dir=save_dir)
            elif 'batched' in self.args.algorithm:
                trainer = BatchedAgentSacTrainer(self.cfg, self.args, global_count, global_writer_loss_count,
                                                 global_writer_quality_count,
                                                 global_win_event_count=global_win_event_count,
                                                 action_stats_count=action_stats_count,
                                                 save_dir=save_dir)
            else:
                trainer = AgentSacTrainer(self.cfg, self.args, global_count, global_writer_loss_count,
                                          global_writer_quality_count, global_win_event_count=global_win_event_count,
                                          action_stats_count=action_stats_count,
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
        else:
            if 'seed_test_sg_global' in self.args.algorithm:
                validator = SacValidate(self.cfg, self.args, save_dir)
                seed, qual = validator.validate_seeds()
                a = 1

