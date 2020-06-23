import os
import torch
from utils.general import Counter
from torch import multiprocessing as mp
from agents.dql import AgentDqlTrainer
import yaml

class TrainDql(object):

    def __init__(self, args, eps=0.9):
        super(TrainDql, self).__init__()
        self.args = args
        self.eps = eps
        self.device = torch.device("cuda:0")

    def train(self, time):
        # Creating directories.
        save_dir = os.path.join(self.args.base_dir, 'results/retrace', self.args.target_dir)
        log_dir = os.path.join(save_dir, 'logs')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if os.path.exists(os.path.join(save_dir, 'runtime_cfg.yaml')):
            os.remove(os.path.join(save_dir, 'runtime_cfg.yaml'))

        print(self.args.pretty())

        with open(os.path.join(save_dir, 'config.txt'), "w") as info:
            info.write(self.args.pretty())

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
            trainer = AgentDqlTrainer(self.args, global_count, global_writer_loss_count,
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

