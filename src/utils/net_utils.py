import os
import pdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.utils import accumulator, utils, io_utils, net_utils

def get_data(data):
    if type(data) == type(list()):
        return [dt.clone().cpu().data for dt in data]
    else:
        return data.clone().cpu().data

def where(cond, x1, x2):
    """ Differentiable equivalent of np.where (or tf.where)
        Note that type of three variables should be same.
    Args:
        cond: condition
        x1: selected value if condition is 1 (True)
        x2: selected value if condition is 0 (False)
    """
    return (cond * x1) + ((1-cond) * x2)

def idx2onehot(idx, num_labels):
    """ Convert indices to onethot vector
    """
    B = idx.size(0)
    one_hot = torch.zeros(B, num_labels)
    one_hot.scatter_(1, idx.view(-1,1), 1)
    return one_hot

def compute_kl_div(inputs, targets, tau=-1, \
                    apply_softmax_on_target=True, reduce=False):
    """
    N = inputs.size(0)
    C = inputs.size(1)

    class_mask = inputs.data.new(N, C).fill_(0)
    class_mask = Variable(class_mask)
    class_mask[targets!=0]=1
    targets[targets==0] = 2 # Avoid log(0)

    probs = (logP.exp()*class_mask).sum(1).view(-1,1)
    batch_loss = (targets * (targets.log() - logP) * class_mask).sum(1)
    """

    if tau > 0:
        inputs = inputs / tau
        if apply_softmax_on_target:
            targets = targets / tau

    logP = F.log_softmax(inputs, dim=1)
    if apply_softmax_on_target:
        targets = F.softmax(targets, dim=1)
    kld_loss = -(targets * logP).mean(dim=1)

    if reduce:
        kld_loss = kld_loss.mean()

    return kld_loss

def compute_margin_loss(logit_list, gt_idx, assigned_idx, margin, reduce=False):
    """ Compute margin loss with logit (or prob) for assigned model
    Args:
        logit_list: list of logits (or prob); m * [B, num_answrs]
        gt_idx: answer label
        assigned_idx: index of assigned model
        margin: margin for distance between prob in gt (assigned)
            and prob in max (not assigned), that is margin should
            be bigger than this value
        use_logit: if true, use logit to compute margin, if not use prob
        reduce: applying average along labels
    """
    # check input correction
    assert type(logit_list) == type(list()), "logits should be given as list type"

    # compute logit of assigned model at ground-truth label
    # P_m_{assigned_idx}(y^*|x)
    B = logit_list[0].size(0)
    gt_logit = logit_list[assigned_idx].gather(1, gt_idx.view(B,1)).squeeze() # [B,]

    margin_loss = 0
    num_models = len(logit_list)
    for mi in range(num_models):
        if mi == assigned_idx:
            continue
        else:
            """
            # compute maximum logit for other models: max(P_m_{others}(y|x))
            # accept that y at maximum logit can be ground-truth
            max_logit, max_idx = logit_list[mi].topk(2, dim=1) # [B, 2]
            max_in_gt = max_idx[:,0].eq(gt_idx).float()
            dist = net_utils.where(max_in_gt,
                    gt_logit - max_logit[:,1], gt_logit - max_logit[:,0])
            """
            # anyway put margin whether y at maximum logit is a ground-truth
            # label
            max_logit, _ = logit_list[mi].max(dim=1) # [B,]
            dist = gt_logit - max_logit
            #dist = gt_logit.log_() - max_logit.log_()

            # add margin loss for model_mi
            cur_loss = (margin - dist).clamp(min=0.0)#.pow(2)
            margin_loss += cur_loss

    if reduce:
        margin_loss = margin_loss.mean()

    return margin_loss

""" DEPRECATED """
def ___compute_logit_margin(logit_list, gt_idx, margin_threshold, reduce=False):

    B = gt_idx.size(0)
    if type(logit_list) == type(list()):
        margins = []
        num_models = len(logit_list)
        for mi in range(num_models):
            gt_logit = logit_list[mi].gather(1, gt_idx.view(B,1))
            diff = (gt_logit.view(B,1)-logit_list[mi]).abs()
            margin = torch.clamp(diff, min=margin_threshold)
            margins.append(margin.mean(dim=1))

        if reduce:
            for mi in range(num_models):
                margins[mi] = margins[mi].mean()
    else:
        gt_logit = logit_list[mi].gather(1, gt_idx.view(B,1))
        margins = torch.clamp(
            gt_logit.view(B,1)-logit_list[mi], min=self.margin_threshold).mean(dim=1)

        if reduce:
            margins = margins.mean()

    return margins
