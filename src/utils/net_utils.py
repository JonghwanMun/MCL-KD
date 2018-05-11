import os
import pdb
import json
import numpy as np
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.utils import accumulator, utils, io_utils, net_utils
from src.externals.VQA_evaluation_APIs.PythonHelperTools.vqaTools.vqa import VQA
from src.externals.VQA_evaluation_APIs.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

def get_data(ptdata, to_clone=True):
    if to_clone:
        if type(ptdata) == type(list()):
            return [dt.clone().cpu().data for dt in ptdata]
        else:
            return ptdata.clone().cpu().data
    else:
        if type(ptdata) == type(list()):
            return [dt.cpu().data for dt in ptdata]
        else:
            return ptdata.cpu().data

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

# Currently not implemented
def compute_oracle_accuracy(logit_list, gts, at):
    assert type(logit_list) == type(list()), \
        "logits should be list() for computing oracle accuracy"

    B = logit_list[0].size(0)
    at = list(range(1, self.num_models+1))
    # compute oracle accuracy
    if self.prob_list == None:
        self.prob_list = [F.softmax(logit, dim=1) for logit in logit_list]
    self.base_top1_predictions = []
    for m in range(self.num_models):
        val, idx = self.prob_list[m].max(dim=1)
        idx = net_utils.get_data(idx, to_clone=False)
        if self.config["misc"]["dataset"] == "vqa":
            self.base_top1_predictions.append(idx)
        if m == 0:
            correct_mask = torch.eq(idx, gts)
        else:
            correct_mask += torch.eq(idx, gts)
    correct_mask = correct_mask.clamp(min=0, max=1)
    num_correct = correct_mask.sum()


def compute_kl_div(inputs, targets, tau=-1, \
                    apply_softmax_on_target=True, reduce=False):

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

def compute_margin_loss(logit_list, gt_idx, assigned_idx, margin, beta, reduce=False):
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

    margin_loss = []
    num_models = len(logit_list)
    for mi in range(num_models):
        if mi == assigned_idx:
            margin_loss.append(0) # we will not access this value
        else:
            """
            # compute maximum logit for other models: max(P_m_{others}(y|x))
            # accept that y at maximum logit can be ground-truth
            max_logit, max_idx = logit_list[mi].topk(2, dim=1) # [B, 2]
            max_in_gt = max_idx[:,0].eq(gt_idx).float()
            diff = net_utils.where(max_in_gt,
                    gt_logit - max_logit[:,1], gt_logit - max_logit[:,0])
            """
            # anyway, difference will be bigger than margin
            # whether y at maximum logit is a ground-truth label or not
            max_logit, _ = logit_list[mi].max(dim=1) # [B,]
            diff = gt_logit - max_logit

            # add margin loss for model_mi
            cur_loss = (margin - diff).clamp(min=0.0)#.pow(2)
            if reduce:
                cur_loss = cur_loss.mean()
            margin_loss.append(beta * cur_loss)

    return margin_loss

def att_fn(logit):
    """ return l2_norm( \sum_c|logit_c|^p ) where p=2
    Args:
        logit: logit from Convolution layer; [B, C, H, W]
    """
    B = logit.size(0)
    return F.normalize(logit.pow(2).mean(1).view(B,-1)) # [B, h*w]

def compute_attention_transfer_loss(student, teacher, beta, reduce=False):
    """ Compute attention transfer loss with logit (or prob) for assigned model
    Args:
        student: list of activations from student network; l * [B,C,H,W]
        teacher: list of activations from teacher network; l * [B,C,H,W]
        beta: multiplier for attention transfer loss
        reduce: applying average over batch
    """
    # check input correction
    assert len(student) == len(teacher), \
        "[AttentionTransferLoss] The number of layers " \
        + "should be same for student and teacher"

    at_loss_list = []
    for s,t in zip(student, teacher):
        at_loss_list.append((att_fn(s) - att_fn(t)).pow(2).mean(1)) # [B,]

    if reduce:
        at_loss_list = [atl.mean() for atl in at_loss_list]

    return beta * sum(at_loss_list)

def get_combinations(n, k):
    """ Return non-overlapped two sets using combinations of nCk
        e.g.
        n=3, k=1 --> return [(0), (1), (2)], [(1,2), (0,2), (1,2)]]
        n=3, k=2 --> return [(0,1), (0,2), (1,2)], [(2), (1), (0)]]
    """
    model_idx = set(np.arange(n))
    assigns = list(combinations(model_idx, k))
    not_assigns = []
    for cbn in assigns:
        not_assigns.append(model_idx - set(cbn))
        assert model_idx == (set.union(set(cbn), not_assigns[-1])), \
            print("Union of assign and not-assign shoud be same with model_idx")

    return assigns, not_assigns

def get_assignment4batch(assigns, idx):
    """ Return assignments for each data
    Args:
        assigns: pre-defined assignment set; nc*[k]
        idx: index of assignment for each item in batch; [B]
    Returns:
        batch_assigns: assignments for batch; [B, max_len]
    """

    B = len(idx)
    k_len = [len(a) for a in assigns]
    max_len = max([len(assigns[i]) for i in idx])
    batch_assigns = -np.ones((B, max_len)) # we use -1 as null assignment
    for i in range(B):
        assign_idx = idx[i]
        batch_assigns[i, 0:k_len[assign_idx]] = assigns[assign_idx]

    return Variable(torch.from_numpy(batch_assigns)).long().t() # [max_k, B]

def assignment2sets(assigns, idx):
    """ Return assignments for each data
    Args:
        assigns: assignment for models over batch; [m,B]
    Returns:
        assign_set: set of assigned model index; [B, set]
        not_assign_set: set of not assigned model index; [B, set]
    """

    B = len(idx)
    k_len = [len(a) for a in assigns]
    max_len = max([len(assigns[i]) for i in idx])
    batch_assigns = np.zeros((B, max_len))
    batch_assigns.fill(-1) # we use -1 as null assignment
    for i in range(B):
        assign_idx = idx[i]
        batch_assigns[i, 0:k_len[assign_idx]] = assigns[assign_idx]

    return Variable(torch.from_numpy(batch_assigns)).long().t() # [max_k, B]

def vqa_evaluate(prediction_json_path, logger, config, modelname="ENS", small_set=True):
    # set up file names and paths
    #ann_path  = "data/VQA_v2.0/annotations/v2_mscoco_val2014_annotations.json"
    #qst_path = "data/VQA_v2.0/annotations/v2_OpenEnded_mscoco_val2014_questions.json"
    if "vqa_eval_annotation_path" in config.keys():
        ann_path  = config["vqa_eval_annotation_path"]
    else:
        ann_path = "data/VQA_v2.0/annotations/v2_mscoco_val2014_annotations.json"
    if "vqa_eval_question_path" in config.keys():
        qst_path = config["vqa_eval_question_path"]
    else:
        qst_path = "data/VQA_v2.0/annotations/v2_OpenEnded_mscoco_val2014_questions.json"

    # create vqa object and vqa_res object
    vqa = VQA(ann_path, qst_path)
    num_all_examples = len(vqa.qa)
    vqa_res = vqa.loadRes(prediction_json_path, qst_path, small_set)
    num_eval_examples = len(vqa.qa)

    # create vqa_eval object by taking vqa and vqa_res
    # n is precision of accuracy (number of places after decimal), default is 2
    vqa_eval = VQAEval(vqa, vqa_res, n=2)

    # evaluate results
    if small_set:
        # If you have a list of question ids on which you would like to evaluate
        # your results, pass it as a list to below function. By default it uses
        # all the question ids in annotation file.
        res_ann = json.load(open(prediction_json_path))
        qst_ids = [int(qst['question_id']) for qst in res_ann]
        acc_per_qstid = vqa_eval.evaluate(qst_ids)
        logger.info("[{}] Accuracy on subset is: {:.02f}".format(
            modelname, vqa_eval.accuracy['overall']))
        logger.info("[{}] Accuracy on all examples is: {:.02f}".format(
              modelname, vqa_eval.accuracy['overall']*num_eval_examples/num_all_examples))
    else:
        acc_per_qstid = vqa_eval.evaluate()
        logger.info("Accuracy is: %.02f" % (vqa_eval.accuracy['overall']))

    return acc_per_qstid


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
