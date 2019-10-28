import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

class CeLoss(nn.Module):

    def __init__(self, weights=None):
        super(CeLoss, self).__init__()
        self.ceLoss = nn.modules.loss.CrossEntropyLoss(weight=weights)


    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        return self.ceLoss(y_pred.flatten(start_dim=2), y_true.flatten(start_dim=2).argmax(1))