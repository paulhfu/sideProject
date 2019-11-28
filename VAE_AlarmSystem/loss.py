import torch
import torch.nn as nn
import torch.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        return

    def forward(self, recon_x, x, mu, logvar, batch_size, img_size, nc):
        # recon_x: image reconstructions
        # x: images
        # mu and logvar: outputs of your encoder
        # batch_size: batch_size
        # img_size: width, respectively height of you images
        # nc: number of image channels
        MSE = F.mse_loss(recon_x, x.view(-1, img_size * img_size * nc))
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalize
        KLD /= batch_size * img_size * img_size * nc
        return MSE + KLD


class DiceLoss(nn.Module):

    def __init__(self, weights=torch.tensor([1, 1], device=device)):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0
        self.weights = weights

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred_l1 = y_pred[:, 0].contiguous().view(-1)
        y_true_l1 = y_true[:, 0].contiguous().view(-1)
        y_pred_l2 = y_pred[:, 1].contiguous().view(-1)
        y_true_l2 = y_true[:, 1].contiguous().view(-1)

        intersection_l1 = (y_pred_l1 * y_true_l1).sum()
        intersection_l2 = (y_pred_l2 * y_true_l2).sum()

        dsc_l1 = (2. * intersection_l1 + self.smooth) / (
            y_pred_l1.sum() + y_true_l1.sum() + self.smooth
        )
        dsc_l2 = (2. * intersection_l2 + self.smooth) / (
                y_pred_l2.sum() + y_true_l2.sum() + self.smooth
        )

        return 1. - self.weights[0] * dsc_l1 - self.weights[1] * dsc_l2

class HypercubeDistLoss(nn.Module):

    def __init__(self, weights=torch.tensor([1, 1], device=device)):
        super(HypercubeDistLoss, self).__init__()
        """  Calculates the distance to the two simplexes in the unit cube defined by the permutations of
            (-.5,.5,.5,.5,...,.5,.5,.5) and (.5,-.5,-.5,-.5,...,-.5,-.5,-.5)
        """
        self.smooth = 1.0
        self.weights = weights

    def forward(self, features):
        for feat in features:
            fgFeat = feat[0]
            bgFeat = feat[1]
            id = torch.flatten(torch.eye(fgFeat.shape[2]))  # flatten generates a view (preserves sparsity of eye)
            invId = torch.flatten(torch.eye(fgFeat.shape[2]))==0.0
            simplex_1 = id - .5
            simplex_2 = invId - .5
            sumDist = torch.tensor([1],dtype=torch.long)
            for fgBn, bgBn in zip(fgFeat, bgFeat):
                # replicate identity vectorwise and features channelwise
                rep1 = nn.functional.pad(fgBn, (fgBn.shape[1], 0), mode='replicate')
                rep2 = nn.functional.pad(simplex_1, (fgBn.shape[0], 0, 0, 0), mode='replicate')
                rep3 = nn.functional.pad(fgBn, (bgBn.shape[1], 0), mode='replicate')
                rep4 = nn.functional.pad(simplex_2, (bgBn.shape[0], 0, 0, 0), mode='replicate')
                pdist = nn.PairwiseDistance(p=2)
                sumDist += self.weights[0] * torch.sum(nn.functional.softmin(pdist(rep1, rep2), dim=1))
                sumDist -= self.weights[0] * torch.sum(nn.functional.softmax(pdist(rep1, rep4), dim=1))
                sumDist += self.weights[1] * torch.sum(nn.functional.softmin(pdist(rep3, rep4), dim=1))
                sumDist -= self.weights[1] * torch.sum(nn.functional.softmax(pdist(rep3, rep2), dim=1))
        return sumDist

class HyperplaneDistLoss(nn.Module):

    def __init__(self, weights=torch.tensor([1, 1], device=device)):
        super(HyperplaneDistLoss, self).__init__()
        """ Loss that calculates the summed distance of fg bg features to each side of the hyperplane 
            <(1,1,1,...,1,1,1),x>+0=0
        """
        self.smooth = 1.0
        self.weights = weights

    def forward(self, features):
        for feat in features:
            fgFeat = feat[0].view([feat[0].shape[0]*feat[0].shape[1]] + [feat[0].shape[2:len(feat[0].shape)]])
            bgFeat = feat[1].view([feat[1].shape[0]*feat[1].shape[1]] + [feat[1].shape[2:len(feat[1].shape)]])
            w = torch.ones(fgFeat.shape)
            b = 0
            norm_w = torch.norm(w)
            sumDist = torch.tensor([1], dtype=torch.long)
            padded_wf = nn.functional.pad(w, (fgFeat.shape[1], 0), mode='replicate')
            padded_wg = nn.functional.pad(w, (bgFeat.shape[1], 0), mode='replicate')
            sumDist += self.weights[0] * (-(torch.einsum('bs,bs->b', fgFeat, padded_wf) + b) / norm_w)
            sumDist += self.weights[1] * (torch.einsum('bs,bs->b', bgFeat, padded_wg) + b) / norm_w
            sumDist += self.weights[0] * torch.sqrt(torch.einsum('bs,bs->b', fgFeat, fgFeat))
            sumDist += self.weights[1] * torch.sqrt(torch.einsum('bs,bs->b', bgFeat, bgFeat))

# class FeatureDistributionLoss(nn.Module):
#     def __init__(self, weights=torch.tensor([1, 1], device=device)):
#         super(HyperplaneDistLoss, self).__init__()
#         """ Loss for difference in feature distributions of query features and support cluster
#         """
#         self.smooth = 1.0
#         self.weights = weights
#
#     def forward(self, features):


class CeLoss(nn.Module):

    def __init__(self, weights=None):
        super(CeLoss, self).__init__()
        self.ceLoss = nn.modules.loss.CrossEntropyLoss(weight=weights)


    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        return self.ceLoss(y_pred.flatten(start_dim=2), y_true.flatten(start_dim=2).argmax(1))