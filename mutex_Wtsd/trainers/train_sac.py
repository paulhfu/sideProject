import os
import torch
import numpy as np
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
from agents.sac_gr_sg import AgentSacTrainer_sg_lg
import yaml
import sys

class TrainSAC(object):

    def __init__(self, cfg):
        super(TrainSAC, self).__init__()
        self.cfg = cfg
        self.device = torch.device("cuda:0")

    def train(self, time):
        # Creating directories.
        save_dir = os.path.join(self.cfg.gen.base_dir, 'results/sac', self.cfg.gen.target_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not self.cfg.gen.evaluate:
            log_dir = os.path.join(save_dir, 'logs')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            if os.path.exists(os.path.join(save_dir, 'runtime_cfg.yaml')):
                os.remove(os.path.join(save_dir, 'runtime_cfg.yaml'))

            print(self.cfg.pretty())

        with open(os.path.join(save_dir, 'config.txt'), "w") as info:
            info.write(self.cfg.pretty())

        if not self.cfg.gen.evaluate:
            rt_cfg_dict = dict(self.cfg.rt_vars)
            cfg_dict = dict(self.cfg.gen)
            cfg_dict.update(dict(self.cfg.fe))
            cfg_dict.update(dict(self.cfg.sac))
            cfg_dict.update(dict(self.cfg.trainer))
            for key in rt_cfg_dict:
                if key in cfg_dict and rt_cfg_dict[key] is None:
                    rt_cfg_dict[key] = cfg_dict[key]
                    self.cfg.rt_vars[key] = cfg_dict[key]
            with open(os.path.join(save_dir, 'runtime_cfg.yaml'), "w") as info:
                yaml.dump(rt_cfg_dict, info)

            global_count = Counter()  # Global shared counter
            global_writer_count = Counter()
            global_writer_loss_count = Counter()  # Global shared counter
            global_writer_quality_count = Counter()  # Global shared counter
            action_stats_count = Counter()

        # ctx = mp.get_context('spawn')
        # q = ctx.Queue()
        # e = ctx.Event()
        # p = ctx.Process(target=send_and_delete_tensors, args=(q, e, torch.cuda.IntTensor, 5))
        # p.start()
        if not self.cfg.gen.evaluate:
            # Start training agents
            manager = mp.Manager()
            return_dict = manager.dict()
            if self.cfg.gen.reward_function == 'defining_rules_lg':
                trainer = AgentSacTrainer_sg_lg(self.cfg, global_count,
                                                         global_writer_loss_count,
                                                         global_writer_quality_count,
                                                         action_stats_count=action_stats_count,
                                                         save_dir=save_dir)
            elif 'seed_test_sg_global' in self.cfg.gen.algorithm:
                trainer = AgentSacTrainer_test_sg_global(self.cfg,
                                                         global_count,
                                                         global_writer_loss_count,
                                                         global_writer_quality_count,
                                                         action_stats_count=action_stats_count,
                                                         global_writer_count=global_writer_count,
                                                         save_dir=save_dir)
            elif 'seed_test_sg' in self.cfg.gen.algorithm:
                trainer = AgentSacTrainer_test_sg(self.cfg, global_count, global_writer_loss_count,
                                                  global_writer_quality_count,
                                                  action_stats_count=action_stats_count,
                                                  save_dir=save_dir)
            elif 'seed_test_global' in self.cfg.gen.algorithm:
                trainer = AgentSacTrainer_test_global(self.cfg, global_count, global_writer_loss_count,
                                                      global_writer_quality_count,
                                                      action_stats_count=action_stats_count,
                                                      save_dir=save_dir)
            elif 'seed_test' in self.cfg.gen.algorithm:
                trainer = AgentSacTrainer_test(self.cfg, global_count, global_writer_loss_count,
                                               global_writer_quality_count,
                                               action_stats_count=action_stats_count,
                                               save_dir=save_dir)
            elif 'global_sparse' in self.cfg.gen.algorithm:
                trainer = AgentSacSRTrainer(self.cfg, global_count, global_writer_loss_count,
                                            global_writer_quality_count,
                                            action_stats_count=action_stats_count,
                                            save_dir=save_dir)
            elif 'batched_sg_rew' in self.cfg.gen.algorithm:
                trainer = BatchedSgRewAgentSacTrainer(self.cfg, global_count, global_writer_loss_count,
                                                      global_writer_quality_count,
                                                      action_stats_count=action_stats_count,
                                                      save_dir=save_dir)
            elif 'batched' in self.cfg.gen.algorithm:
                trainer = BatchedAgentSacTrainer(self.cfg, global_count, global_writer_loss_count,
                                                 global_writer_quality_count,
                                                 action_stats_count=action_stats_count,
                                                 save_dir=save_dir)
            else:
                trainer = AgentSacTrainer(self.cfg, global_count, global_writer_loss_count,
                                          global_writer_quality_count,
                                          action_stats_count=action_stats_count,
                                          save_dir=save_dir)

            rns = torch.randint(0, 2 ** 32, torch.Size([10]))
            best_qual = -np.inf
            best_seed = None
            for i, rn in enumerate(rns):
                # Start validation agent
                processes = []
                for rank in range(self.cfg.gen.n_processes_per_gpu * self.cfg.gen.n_gpu):
                    if rank == 0:
                        p = mp.Process(target=trainer.train, args=(rank, time, return_dict, rn.item()))
                    else:
                        p = mp.Process(target=trainer.train, args=(rank, time, None, rn.item()))
                    p.start()
                    processes.append(p)
                # Clean up
                for p in processes:
                    p.join()

                if return_dict['score'] > best_qual:
                    best_qual = return_dict['score']
                    best_seed = rn.item()

                # sys.stdout = open("/dev/stdout", "w")
                res = 'best seed is: ' + str(best_seed) + " with a qual of: " + str(best_qual)
                print(res)

            with open(os.path.join(save_dir, 'result.txt'), "w") as info:
                info.write(res)

            if self.cfg.gen.cross_validate_hp or self.cfg.gen.test_score_only:
                print('Score is: ', return_dict['test_score'])
                print('Wins out of 20 trials')
                return return_dict['test_score']
        else:
            if 'seed_test_sg_global' in self.cfg.gen.algorithm:
                validator = SacValidate(self.cfg.sac, self.cfg.gen, save_dir)
                seed, qual = validator.validate_seeds()
                a = 1

