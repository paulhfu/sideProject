import torch
from data.disjoint_discs import MultiDiscSpGraphDset
from torch.optim import Adam
from models.sp_embed_unet import SpVecsUnet
from torch.utils.data import DataLoader

from utils.reward_functions import GraphDiceLoss
from agents.replayMemory import TransitionData
from collections import namedtuple
from mu_net.criteria.contrastive_loss import ContrastiveLoss
from environments.sp_grph_gcn_1 import SpGcnEnv
from models.GCNNs.q_value_net import GcnEdgeAngle1dQ
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from utils.general import adjust_learning_rate
import os
import yaml

class TrainNaiveGcn(object):

    def __init__(self, args, eps=0.9):
        super(TrainNaiveGcn, self).__init__()
        self.args = args
        self.device = torch.device("cuda:0")
        self.setup()

    def setup(self):
        # Creating directories.
        self.save_dir = os.path.join(self.args.base_dir, 'results/naive_gcn', self.args.target_dir)
        log_dir = os.path.join(self.save_dir, 'logs')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if os.path.exists(os.path.join(self.save_dir, 'config.yaml')):
            os.remove(os.path.join(self.save_dir, 'config.yaml'))
        print(' ' * 26 + 'Options')

        # Saving parameters
        with open(os.path.join(self.save_dir, 'params.txt'), 'w') as f:
            for k, v in vars(self.args).items():
                print(' ' * 26 + k + ': ' + str(v))
                f.write(k + ' : ' + str(v) + '\n')

        with open(os.path.join(self.save_dir, 'config.yaml'), "w") as info:
            documents = yaml.dump(vars(self.args), info)

        torch.manual_seed(self.args.seed)

    def train(self):
        step_counter = 0
        device = torch.device("cuda:" + str(0))
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

        # Create shared network
        model = GcnEdgeAngle1dQ(self.args.n_raw_channels, self.args.n_embedding_features,
                                self.args.n_edge_features, 1, device, writer=writer)

        if self.args.no_fe_extr_optim:
            for param in model.fe_ext.parameters():
                param.requires_grad = False

        model.cuda(device)
        dloader = DataLoader(MultiDiscSpGraphDset(no_suppix=False), batch_size=1, shuffle=True, pin_memory=True,
                             num_workers=0)
        optimizer = Adam(model.parameters(), lr=self.args.lr)
        loss = GraphDiceLoss()

        if self.args.fe_extr_warmup and not self.args.test_score_only:
            fe_extr = SpVecsUnet(self.args.n_raw_channels, self.args.n_embedding_features, device)
            fe_extr.cuda(device)
            self.fe_extr_warm_start(fe_extr, writer=writer)
            model.fe_ext.load_state_dict(fe_extr.state_dict())
            if self.args.model_name == "":
                torch.save(model.state_dict(), os.path.join(self.save_dir, 'agent_model'))
            else:
                torch.save(model.state_dict(), os.path.join(self.save_dir, self.args.model_name))

        if self.args.model_name != "":
            model.load_state_dict(torch.load(os.path.join(self.save_dir, self.args.model_name)))
        elif self.args.fe_extr_warmup:
            print('loaded fe extractor')
            model.load_state_dict(torch.load(os.path.join(self.save_dir, 'agent_model')))

        while step_counter <= self.args.T_max:
            if step_counter == 78:
                a = 1
            if (step_counter + 1) % 1000 == 0:
                post_input = True
            else:
                post_input = False
            with open(os.path.join(self.save_dir, 'config.yaml')) as info:
                args_dict = yaml.full_load(info)
                if args_dict is not None:
                    if 'lr' in args_dict:
                        self.args.lr = args_dict['lr']
                        adjust_learning_rate(optimizer, self.args.lr)

            round_n = 0

            raw, gt, sp_seg, sp_indices, edge_ids, edge_weights, gt_edges, edge_features = \
                self._get_data(dloader, device)

            inp = [obj.float().to(model.device) for obj in [edge_weights, sp_seg, raw + gt, sp_seg]]
            pred, side_loss = model(inp, sp_indices=sp_indices,
                                    edge_index=edge_ids.to(model.device),
                                    angles=None,
                                    edge_features_1d=edge_features.to(model.device),
                                    round_n=round_n, post_input=post_input)

            pred = pred.squeeze()


            loss_val = loss(pred, gt_edges.to(device))

            ttl_loss = loss_val + side_loss
            quality = (pred - gt_edges.to(device)).abs().sum()

            optimizer.zero_grad()
            ttl_loss.backward()
            optimizer.step()

            if writer is not None:
                writer.add_scalar("step/lr", self.args.lr, step_counter)
                writer.add_scalar("step/dice_loss", loss_val.item(), step_counter)
                writer.add_scalar("step/side_loss", side_loss.item(), step_counter)
                writer.add_scalar("step/quality", quality.item(), step_counter)

            step_counter += 1

        a=1



    def fe_extr_warm_start(self, sp_feature_ext, writer=None):
        dataloader = DataLoader(MultiDiscSpGraphDset(length=self.args.fe_warmup_iterations * 10), batch_size=10,
                                shuffle=True, pin_memory=True)
        criterion = ContrastiveLoss(delta_var=0.5, delta_dist=1.5)
        optimizer = torch.optim.Adam(sp_feature_ext.parameters())
        writer_idx_warmup_loss = 0
        for i, (data, gt) in enumerate(dataloader):
            data, gt = data.to(sp_feature_ext.device), gt.to(sp_feature_ext.device)
            pred = sp_feature_ext(data)
            loss = criterion(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if writer is not None:
                writer.add_scalar("loss/fe_warm_start", loss.item(), writer_idx_warmup_loss)
                writer_idx_warmup_loss += 1


    def _get_data(self, dloader, device):
        edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes, affinities, gt = \
            next(iter(dloader))

        edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, raw, nodes, affinities, gt = \
            edges.squeeze(), edge_feat.squeeze(), diff_to_gt.squeeze(), \
            gt_edge_weights.squeeze(), node_labeling.squeeze(), raw.squeeze(), nodes.squeeze(), \
            affinities.squeeze().numpy(), gt.squeeze()

        edge_weights = edge_feat[:, 0].squeeze().to(device)

        stacked_superpixels = [node_labeling == n for n in nodes]
        sp_indices = [sp.nonzero().to(device) for sp in stacked_superpixels]

        return raw, gt, node_labeling, sp_indices, edges, edge_weights, gt_edge_weights, edge_feat
