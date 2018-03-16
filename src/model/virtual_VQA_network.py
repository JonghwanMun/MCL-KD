import os
import pdb
import copy
import json
import logging
import numpy as np
from collections import OrderedDict
from sklearn.metrics import confusion_matrix as cnf_mat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from src.model import building_blocks
from src.model.virtual_network import VirtualNetwork
from src.utils import accumulator, utils, io_utils, vis_utils, net_utils

class VirtualVQANetwork(VirtualNetwork):
    def __init__(self, config):
        # update options compatible for a specific VQA model
        config = self.model_specific_config_update(config)

        # save configuration for later network reproduction
        save_config_path = os.path.join(
            config["misc"]["result_dir"], "config.yml")
        io_utils.write_yaml(save_config_path, config)
        self.config = config

        # Must call super __init__()
        super(VirtualVQANetwork, self).__init__()

        # print configuration
        self.logger["train"].info(json.dumps(config, indent=2))

    def _set_sample_data(self, data):
        if self.sample_data == None:
            self.sample_data = copy.deepcopy(data)

    def reset_status(self):
        """ Reset (initialize) metric scores or losses (status).
        """
        if self.status == None:
            self.status = OrderedDict()
            self.status["loss"] = 0
            self.status["top1"] = 0
        else:
            for k in self.status.keys():
                self.status[k] = 0

        self.all_predictions = []

    def tensor2variable(self, tensors):
        """ Convert tensors to Variable
        Args:
            tensors: list of [inputs for network, img info]; inputs for network
        Returns:
            Variables: list of Variables; inputs for network
        """
        tensors = tensors[0]
        variables = []
        variables.append(Variable(tensors[0]).cuda()
                         if self.use_gpu else Variable(tensors[0]))
        variables.append(Variable(tensors[1]).cuda()
                         if self.use_gpu else Variable(tensors[1]))
        variables.append(tensors[2])
        if len(tensors) == 5:
            variables.append(Variable(tensors[-2], requires_grad=False).cuda() \
                             if self.use_gpu
                             else Variable(tensors[-2], requires_grad=False))
        if type(tensors[-1]) == type(list()):
            variables.append([Variable(ts).cuda() \
                              if self.use_gpu else Variable(ts) \
                              for ts in tensors[-1]])
        else:
            variables.append(Variable(tensors[-1]).cuda()
                             if self.use_gpu else Variable(tensors[-1]))

        return variables

    """ methods for checkpoint """
    def load_checkpoint(self, ckpt_path):
        """ Load checkpoint of the network.
        Args:
            ckpt_path: checkpoint file path
        """
        model_state_dict = torch.load(ckpt_path)
        for m in model_state_dict.keys():
            if m in self.model_list:
                self[m].load_state_dict(model_state_dict[m])
            else:
                self.logger["train"].info("{} is not in {}".format(
                        m, "|".join(self.model_list)))

        self.logger["train"].info("[{}] are initialized from checkpoint ({})".format(
                "|".join(model_state_dict.keys()), ckpt_path))

    def save_checkpoint(self, cid):
        """ Save checkpoint of the network.
        Args:
            cid: id of checkpoint
        """
        ckpt_path = os.path.join(self.config["misc"]["result_dir"], \
                "checkpoints", "checkpoint_epoch_{:03d}.pkl")
        model_state_dict = OrderedDict()
        for m in self.model_list:
            model_state_dict[m] = self[m].state_dict()
        torch.save(model_state_dict, ckpt_path.format(cid))
        self.logger["train"].info("Checkpoint is saved in {}".format(
                ckpt_path.format(cid)))

    def attach_assignments(self):
        if self.config["model"]["version"] != "IE":
            self.qst_ids_list.append(self.cur_qids)
            self.assignments_list.append( \
                    self.criterion.assignments.clone().numpy())

    def attach_predictions(self):
        for i in range(len(self.cur_qids)):
            if isinstance(self.cur_qids[i], int):
                qid = int(self.cur_qids[i])
            elif ("train" in self.cur_qids[i]) or ("val" in self.cur_qids[i]) \
                    or ("test" in self.cur_qids[i]):
                qid = self.cur_qids[i]

            self.all_predictions.append({
                "question_id": qid,
                "answer": utils.label2string(self.itoa, self.top1_predictions[i])
            })

    def save_predictions(self, prefix):
        save_dir = os.path.join(self.config["misc"]["result_dir"], "predictions")
        io_utils.write_json(os.path.join(
            save_dir, prefix+".json"), self.all_predictions)

    """ methods for counters """
    def _create_counters(self):
        self.counters = OrderedDict()
        self.counters["loss"] = accumulator.Accumulator("loss")
        self.counters["top1"] = accumulator.Accumulator("top1")

    def apply_curriculum_learning(self):
        return

    def bring_loader_info(self, dataset):
        if type(dataset) == type(dict()):
            self.itow = dataset["train"].get_itow()
            self.itoa = dataset["train"].get_itoa()
            self.fetching_answer_option = dataset["train"].fetching_answer_option
            self.origin_train_qst_ids = dataset["train"].get_qst_ids()
            self.origin_test_qst_ids = dataset["test"].get_qst_ids()
        else:
            self.itow = dataset.get_itow()
            self.itoa = dataset.get_itoa()
            self.fetching_answer_option = dataset.fetching_answer_option
            self.origin_train_qst_ids = dataset.get_qst_ids()
            self.origin_test_qst_ids = dataset.get_qst_ids()

        if self.fetching_answer_option == "all_answers":
            loss_reduce = utils.get_value_from_dict(
                self.config["model"], "loss_reduce", True)
            self.criterion = building_blocks.MultipleCriterion(
                nn.CrossEntropyLoss(), loss_reduce)

    """ methods for visualization """
    def visualize_confusion_matrix(self, epoch, prefix="train"):
        confusion_matrix_list = []
        for ii in range(self.num_models):
            cnf = cnf_mat(np.hstack(self.gt_list), np.hstack(self.base_pred_all_list[ii]))
            confusion_matrix_list.append(cnf)

        # save results of CMCL (assignment of model)
        class_names = [self.itoa[str(key)] for key in range(len(self.itoa.keys()))]
        vis_utils.save_confusion_matrix_visualization(
                self.config, confusion_matrix_list, class_names, epoch, prefix)

    """ methods for metrics """
    def compute_confusion_matrix(self, logit_list, gts):
        """ Compute confusion matrix for each model
        Args:
            logit_list: list of m * [batch_size, num_answers]
            gts: ground-truth answers [batch_size]
        """
        self.basenet_pred = []
        B = logit_list[0].size(0)
        prob_list = [F.softmax(logit, dim=1) for logit in logit_list]
        for ii in range(len(prob_list)):
            val, idx = prob_list[ii].max(dim=1)
            self.base_pred_all_list[ii].append(idx.data.clone().cpu().numpy())
            self.basenet_pred.append(utils.label2string(self.itoa, idx.data.cpu()[0]))
        self.gt_list.append(gts.clone().cpu().numpy())

    def compute_top1_accuracy(self, logits, gts):
        """ Compute Top-1 Accuracy
        Args:
            logits list of m * [batch_size, num_answers]
            gts: ground-truth answers [batch_size]
        """
        if type(logits) == type(list()):
            # compute probabilities and average them
            B = logits[0].size(0)
            prob_list = [F.softmax(logit, dim=1) for logit in logits]
            probs = torch.stack([prob for prob in prob_list], 0) # [m, B, num_answers]
            probs = torch.mean(probs, dim=0) # [B, num_answers]

        else:
            # compute probabilities and average them
            B = logits.size(0)
            probs = F.softmax(logits, dim=1)
            probs = net_utils.get_data(probs)

        # count the number of correct predictions
        val, max_idx = net_utils.get_data(probs).max(dim=1)
        num_correct = torch.sum(torch.eq(max_idx, gts))

        # save top1 related status
        self.status["top1"] = num_correct / B
        self.top1_predictions = max_idx
        self.top1_gt = utils.label2string(self.itoa, gts[0])
        self.counters["top1"].add(num_correct, B)

        # count the number of correct predictions and save them for each model
        if self.config["misc"]["model_type"] == "ensemble" \
                and self.config["model"]["verbose_all"]:
            assert type(logits) == type(list()),\
                "To verbose all, you should perform ensemble"
            for m in range(self.num_models):
                val, idx = prob_list[m].data.cpu().max(dim=1)
                num_correct = torch.sum(torch.eq(idx, gts))
                model_name = "M{}".format(m)
                self.status[model_name] = num_correct / B
                self.counters[model_name].add(num_correct, B)

    def compute_oracle_accuracy(self, logit_list, gts):
        """ Compute Oracle Accuracy
        Args:
            logit_list: list of m * [batch_size, num_answers]
            gts: ground-truth answers [batch_size]
        """
        assert type(logit_list) == type(list()), \
            "logits should be list() for computing oracle accuracy"

        B = logit_list[0].size(0)
        # compute oracle accuracy
        prob_list = [F.softmax(logit, dim=1) for logit in logit_list]
        for ii in range(len(prob_list)):
            val, idx = prob_list[ii].max(dim=1)
            if ii == 0:
                correct_mask = torch.eq(idx.cpu().data, gts)
            else:
                correct_mask += torch.eq(idx.cpu().data, gts)
        correct_mask = correct_mask.clamp(min=0, max=1)
        num_correct = correct_mask.sum()

        self.status["oracle"] = num_correct / B
        self.counters["oracle"].add(num_correct, B)

    def compute_assignment_accuracy(self, logit_list, gts,
                                   assignment_logits=None, hard=True):
        """ Compute Accuracy using assignment layer
        Args:
            logit_list: list of m * [batch_size, num_answers]
            gts: ground-truth answers [batch_size]
            assignment_logits: logits for model assignment
        """
        assert type(logit_list) == type(list()), \
            "logits should be list() for computing predicted assignment-based accuracy"

        B = logit_list[0].size(0)
        if type(assignment_logits) is type(None):
            self.status["sel-acc"] = 0
            self.counters["sel-acc"].add(0, B)
            return

        # obtain indices of the highest probability
        prob_list = [F.softmax(logit, dim=1) for logit in logit_list]
        idx_list = []
        for ii in range(len(prob_list)):
            val, idx = prob_list[ii].max(dim=1)
            idx_list.append(idx.data.clone().cpu().numpy())

        # compute model assignment
        assignment_logits = assignment_logits.data.clone().cpu()
        _, assignments = assignment_logits.max(dim=1)

        # count the correct numbers
        num_correct = 0
        if hard:
            for bi in range(B):
                if gts[bi] == idx_list[assignments[bi]][bi]:
                    num_correct += 1
        else:
            B, m = assignment_logits.size()
            prob_tensor = torch.stack([prob.data.cpu() for prob in prob_list], 0) # [m, B, num_answers]
            prob_tensor = prob_tensor * assignment_logits.t().contiguous().view(m,B,1).expand_as(prob_tensor)
            prob_tensor = torch.mean(prob_tensor, dim=0) # [B, num_answers]

            # count the number of correct predictions
            val, idx = prob_tensor.max(dim=1)
            num_correct = torch.sum(torch.eq(idx, gts))

        self.status["sel-acc"] = num_correct / B
        self.counters["sel-acc"].add(num_correct, B)
        self.pred_assignments = assignments

    def compute_gt_assignment_accuracy(self, logit_list, gts, assignments=None):
        """ Compute Accuracy using assignments with gt
        Args:
            logit_list: list of m * [batch_size, num_answers]
            gts: ground-truth answers [batch_size]
        """
        assert type(logit_list) == type(list()), \
            "logits should be list() for computing gt assignment-based accuracy"
        B = logit_list[0].size(0)
        if not "gt-sel-acc" in self.status.keys():
            self.status["gt-sel-acc"] = 0
            self.counters["gt-sel-acc"] = accumulator.Accumulator("gt-sel-acc")
        if type(assignments) is type(None):
            self.status["gt-sel-acc"] = 0
            self.counters["gt-sel-acc"].add(0, B)
            return

        if assignments.size(1) > 1:
            assignments = assignments[:, 0]
        assignments = assignments.squeeze()
        # obtain indices of the highest probability
        prob_list = [F.softmax(logit, dim=1) for logit in logit_list]
        idx_list = []
        for ii in range(len(prob_list)):
            val, idx = prob_list[ii].max(dim=1)
            idx_list.append(idx.data.clone().cpu().numpy())

        # count the correct numbers
        num_correct = 0
        for bi in range(B):
            if gts[bi] == idx_list[assignments[bi]][bi]:
                num_correct += 1

        self.status["gt-sel-acc"] = num_correct / B
        self.counters["gt-sel-acc"].add(num_correct, B)

    """ method for status (metrics) """
    def compute_status(self, logits, gts):
        if type(gts) == type(list()):
            gts = gts[0] # we use most frequent answers for vqa

        self.compute_top1_accuracy(logits, gts)
        self.attach_predictions()

    def print_status(self, epoch, iteration, prefix="",
                     mode="train", is_main_net=True):
        """ Print status (scores, etc)
        Args:
            epoch: current epoch
            iteration: current iteration
            prefix: identity to distinguish models;
                    if is_main_net, this is not needed
            is_main_net: flag about this network is root
        """

        if self.config["misc"]["model_type"] == "saaa":
            self.saaa.print_status(self.logger["train"], prefix)
        elif self.config["misc"]["model_type"] == "san":
            self.san.print_status(self.logger["train"], prefix)

        if is_main_net:
            # prepare txt to print
            txt = "epoch {} step {}".format(epoch, iteration)
            for k,v in self.status.items():
                txt += ", {} = {:.3f}".format(k, v)
            txt += ", ({}|{})".format(utils.label2string(
                        self.itoa, self.top1_predictions[0]), self.top1_gt)

            # print learning information
            if mode == "train":
                self.logger["train"].info(txt)
            else:
                self.logger["train"].debug(txt)

    def set_is_main_net(self, is_main_net):
        self.is_main_net = is_main_net

    @classmethod
    def model_specific_config_update(cls, config):
        print("You would need to implement 'model_specific_config_update'")
        return config

    @staticmethod
    def override_config_from_loader(config, loader):
        # model: question embedding layer
        config["model"]["vocab_size"] = loader.get_vocab_size()
        config["model"]["word_emb_padding_idx"] = loader.get_idx_empty_word()
        # model: classifier and assignment layer
        config["model"]["num_labels"] = loader.get_num_answers()

        return config

    @staticmethod
    def override_config_from_params(config, params):
        config = VirtualNetwork.override_config_from_params(config, params)

        return config
