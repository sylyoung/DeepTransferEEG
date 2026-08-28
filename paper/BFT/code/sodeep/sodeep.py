# Differentiable sorting utilities.  SpearmanLoss below is the objective used
# to train the ranking module r, i.e. Eq. (6) of the paper
# "Backpropagation-Free Test-Time Adaptation for Lightweight EEG-Based BCIs"
# (IEEE J. Biomed. Health Inform., 2026).
import torch

from model import model_loader
from sodeep_utils import get_rank


def load_sorter(checkpoint_path):
    # sorter_checkpoint = torch.load(checkpoint_path)
    sorter_checkpoint = torch.load(checkpoint_path, weights_only=False)

    model_type = sorter_checkpoint["args_dict"].model_type
    seq_len = sorter_checkpoint["args_dict"].seq_len
    state_dict = sorter_checkpoint["state_dict"]

    return model_type, seq_len, state_dict


class RankHardLoss(torch.nn.Module):
    """ Loss function  inspired by hard negative triplet loss, directly applied in the rank domain """
    def __init__(self, sorter_type, seq_len=None, sorter_state_dict=None, margin=0.2, nmax=1):
        super(RankHardLoss, self).__init__()
        self.nmax = nmax
        self.margin = margin

        self.sorter = model_loader(sorter_type, seq_len, sorter_state_dict)

    def hc_loss(self, scores):
        rank = self.sorter(scores)

        diag = rank.diag()

        rank = rank + torch.diag(torch.ones(rank.diag().size(), device=rank.device) * 50.0)

        sorted_rank, _ = torch.sort(rank, 1, descending=False)

        hard_neg_rank = sorted_rank[:, :self.nmax]

        loss = torch.sum(torch.clamp(-hard_neg_rank + (1.0 / (scores.size(1)) + diag).view(-1, 1).expand_as(hard_neg_rank), min=0))

        return loss

    def forward(self, scores):
        """ Expect a score matrix with scores of the positive pairs are on the diagonal """
        caption_loss = self.hc_loss(scores)
        image_loss = self.hc_loss(scores.t())

        image_caption_loss = caption_loss + image_loss

        return image_caption_loss


class RankLoss(torch.nn.Module):
    """ Loss function  inspired by recall """
    def __init__(self, sorter_type, seq_len=None, sorter_state_dict=None,):
        super(RankLoss, self).__init__()
        self.sorter = model_loader(sorter_type, seq_len, sorter_state_dict)

    def forward(self, scores):
        """ Expect a score matrix with scores of the positive pairs are on the diagonal """
        caption_rank = self.sorter(scores)
        image_rank = self.sorter(scores.t())

        image_caption_loss = torch.sum(caption_rank.diag()) + torch.sum(image_rank.diag())

        return image_caption_loss


class MapRankingLoss(torch.nn.Module):
    """ Loss function  inspired by mean Average Precision """
    def __init__(self, sorter_type, seq_len=None, sorter_state_dict=None):
        super(MapRankingLoss, self).__init__()

        self.sorter = model_loader(sorter_type, seq_len, sorter_state_dict)

    def forward(self, output, target):
        # Compute map for each classes
        map_tot = 0
        for c in range(target.size(1)):
            gt_c = target[:, c]

            if torch.sum(gt_c) == 0:
                continue
            rank_pred = self.sorter(output[:, c].unsqueeze(0)).view(-1)
            rank_pos = rank_pred * gt_c

            map_tot += torch.sum(rank_pos)

        return map_tot


class SpearmanLoss(torch.nn.Module):
    """ Loss function  inspired by spearmann correlation.self
    Required the trained model to have a good initlization.

    Set lbd to 1 for a few epoch to help with the initialization.
    """
    def __init__(self, sorter_type, seq_len=None, sorter_state_dict=None, lbd=0):
        super(SpearmanLoss, self).__init__()
        self.sorter = model_loader(sorter_type, seq_len, sorter_state_dict)

        self.criterion_mse = torch.nn.MSELoss()
        self.criterionl1 = torch.nn.L1Loss()

        self.lbd = lbd

    def forward(self, mem_pred, mem_gt, pr=False):
        # mem_pred is w_i of Eq. (5), mem_gt the task-loss profile pi_i.
        # rank_pred = m(w_i) is the mapping module applied to the predicted
        # profile, rank_gt the true rank of the target profile.
        #
        # NOTE ON lbd: the loss below is a convex combination, so the DEFAULT
        # lbd = 0 reduces it to L1(w_i, pi_i) and the mapping module m, though
        # evaluated on the line above, contributes nothing to the gradient.
        # Eq. (6) of the paper is the rank-space term, which is only active at
        # lbd = 1 (and is an MSE there, whereas Eq. (6) writes an L1 norm).
        # Every call site in BFT-classify and BFT-regression uses the default.
        rank_gt = get_rank(mem_gt)

        rank_pred = self.sorter(mem_pred.unsqueeze(
            0)).view(-1)

        loss  = self.lbd*self.criterion_mse(rank_pred, rank_gt) + (1 - self.lbd)*self.criterionl1(mem_pred, mem_gt)
        return loss
    
    def get_rank_data(self, mem_pred, mem_gt, pr=False):
        # get_rank: input: [1, 2, 4, 5]
        #           output: [1.0000, 0.7500, 0.5000, 0.2500]
        rank_gt = get_rank(mem_gt)
        rank_pred = self.sorter(mem_pred.unsqueeze(0)).view(-1)
        return rank_pred
