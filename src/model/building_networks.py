import os
import pdb
import copy
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.model import building_blocks
from src.model.virtual_network import VirtualNetwork
from src.model.virtual_VQA_network import VirtualVQANetwork
from src.experiment import common_functions as cmf
from src.utils import accumulator, utils, io_utils, vis_utils, net_utils

class Ensemble(VirtualVQANetwork):
    def __init__(self, config):
        super(Ensemble, self).__init__(config) # Must call super __init__()
        self.classname = "ENSEMBLE"

        # options for loading model
        self.base_model_type = utils.get_value_from_dict(
            config["model"], "base_model_type", "saaa")
        self.M = cmf.get_model(self.base_model_type)
        self.use_gpu = utils.get_value_from_dict(
            config["model"], "use_gpu", True)
        self.num_models = utils.get_value_from_dict(
            config["model"], "num_models", 3)

        # options if use knowledge distillation
        self.use_knowledge_distillation = \
            utils.get_value_from_dict(
                config["model"], "use_knowledge_distillation", False)
        self.use_initial_assignment = \
            utils.get_value_from_dict(
                config["model"], "use_initial_assignment", False)
        base_model_ckpt_path = \
                utils.get_value_from_dict(
                    config["model"], "base_model_ckpt_path", "None")
        if self.use_knowledge_distillation:
            assert base_model_ckpt_path != "None", \
                "checkpoint path for base model should be given"

        # options if use assignment model
        self.use_assignment_model = utils.get_value_from_dict(
            config["model"], "use_assignment_model", False)

        # build base models if use
        if self.use_knowledge_distillation:
            num_base_models = len(base_model_ckpt_path)
            self.base_model = []
            for i in range(num_base_models):
                base_config = copy.deepcopy(config)
                base_config["use_knowledge_distillation"] = False
                self.base_model.append(self.M(base_config))
                self.base_model[i].load_checkpoint(base_model_ckpt_path[i])
                if self.use_gpu and torch.cuda.is_available():
                    self.base_model[i].cuda()
                self.base_model[i].eval() # set to eval mode for base model
                self.logger["train"].info( \
                        "{}th base-net is initialized from {}".format( \
                        i, base_model_ckpt_path[i]))

        # build specialized models
        self.net_list = nn.ModuleList()
        for m in range(self.num_models):
            self.net_list.append(self.M(config))

            # load pre-trained base models if exist
            if base_model_ckpt_path != "None":
                self.net_list[m].load_checkpoint(base_model_ckpt_path[m])
                self.logger["train"].info("{}th net is initialized from {}".format( \
                        m, base_model_ckpt_path[m]))
        if self.use_assignment_model:
            self.assignment_model = \
                    building_blocks.AssignmentModel(config["model"])
        self.criterion = building_blocks.EnsembleLoss(config["model"])

        # set is_main_net flag of base networks as False
        for m in range(self.num_models):
            self.net_list[m].set_is_main_net(False)

        # set models to update
        self.model_list = ["net_list", "criterion"]
        self.models_to_update = ["net_list", "criterion"]
        if self.use_assignment_model:
            self.model_list.append("assignment_model")
            self.models_to_update.append("assignment_model")

        # create status and counter for each model
        if self.config["model"]["verbose_all"]:
            for m in range(self.num_models):
                self.status["M{}".format(m)] = 0
            for m in range(self.num_models):
                model_name = "M{}".format(m)
                self.counters[model_name] = accumulator.Accumulator(model_name)

    def forward(self, data):
        """ Forward network
        Args:
            data: list of two components [inputs for network, image information]
        Returns:
            criterion_inp: input list for criterion
        """
        # forward network
        self.logit_list = []
        ce_loss_list = []
        for m in range(self.num_models):
            ith_net_outputs = self.net_list[m](data)
            ith_net_loss = self.net_list[m].loss_fn(ith_net_outputs[0], data[-1])
            self.logit_list.append(ith_net_outputs[0])
            ce_loss_list.append(ith_net_loss)

        # save inputs for criterion
        criterion_inp = [[self.logit_list, ce_loss_list]]
        self.assignment_logits = None
        if self.use_assignment_model and \
                (self.config["model"]["version"] != "IE"):
            self.assignment_logits = self.assignment_model([data[0], data[1], data[2]])
            criterion_inp[0].append(self.assignment_logits)

        if self.use_initial_assignment and (len(data) == 5):
            criterion_inp[0].append(data[3])

        if self.use_knowledge_distillation:
            self.base_outs = []
            for i in range(len(self.base_model)):
                self.base_outs.append(self.base_model[i].forward(data)[0]) # [logit, attention]
            criterion_inp[0].append(self.base_outs)

        return criterion_inp

    """ Visualization related methods """
    def _set_sample_data(self, data):
        """ Set samples data recursively
        Args:
            data: sample data
        """
        if self.sample_data == None:
            self.sample_data = copy.deepcopy(data)
            for m in range(self.num_models):
                self.net_list[m]._set_sample_data(data)

    def save_assignments(self, prefix, mode="train"):
        assignments = np.vstack(self.assignments_list)
        qst_ids = []
        for qid in self.qst_ids_list:
            qst_ids.extend(qid)
        print("shape of assignments: ", assignments.shape)
        if mode == "train":
            origin_qst_ids = self.origin_train_qst_ids
        else:
            origin_qst_ids = self.origin_test_qst_ids

        assignments, qst_ids = \
            cmf.reorder_assignments_using_qst_ids(origin_qst_ids, qst_ids, assignments, is_subset=True)

        # setting directory for saving assignments
        save_dir = os.path.join(self.config["misc"]["result_dir"], "assignments", mode)
        io_utils.check_and_create_dir(save_dir)

        # save assignments
        save_hdf5_path = os.path.join(save_dir, prefix + "_assignment.h5")
        hdf5_file = io_utils.open_hdf5(save_hdf5_path, "w")
        hdf5_file.create_dataset("assignments", dtype="int32", data=assignments)
        print("Selctions are saved in {}".format(save_hdf5_path))

        # save assignments of qst_ids
        save_json_path = os.path.join(save_dir, prefix + "_qst_ids.json")
        out = {}
        out["question_ids"] = qst_ids
        io_utils.write_json(save_json_path, out)
        print ("Saving is done: {}".format(save_json_path))

    def save_results(self, data, prefix, mode="train"):
        """ Visualize results (attention weights) and save them
        Args:
            data: list [imgs, qst_labels, qst_lenghts, answer_labels, img_path]
        """
        # save predictions
        self.save_predictions(prefix)
        # save visualization of confusion matrix for each model
        epoch = int(prefix.split("_")[-1])
        self.visualize_confusion_matrix(epoch, prefix=mode)

        if mode == "train":
            # maintain sample data
            self._set_sample_data(data)

            # convert data as Variables
            data = [*self.tensor2variable(data), data[1]]

            if self.config["model"]["vis_individual_net"]:
                # save visualization of base network
                # for SAN, we visualize attention weights
                logit_list = []
                loss_list = []
                for m in range(self.num_models):
                    self.net_list[m].save_results(
                        data, "{}_m{}-att".format(prefix, m+1), compute_loss=True)
                    logit_list.append(self.net_list[m].logits)
                    loss_list.append(self.net_list[m].loss)

                criterion_inp = [logit_list, loss_list]
                self.assignment_logits = None
                if self.use_assignment_model and \
                        (self.config["model"]["version"] != "IE"):
                    self.assignment_logits = self.assignment_model( \
                            [data[0], data[1], data[2]])
                    criterion_inp.append(self.assignment_logits)

                if self.use_initial_assignment:
                    criterion_inp.append(data[3])

                if self.use_knowledge_distillation:
                    if self.use_knowledge_distillation:
                        self.base_outs = []
                        for i in range(len(self.base_model)):
                            self.base_outs.append(self.base_model[i](data)[0]) # [logit, attention]
                        criterion_inp.append(self.base_outs)
                outputs = [criterion_inp]
            else:
                outputs = self.forward(data[:-1])
                logit_list = outputs[0][0]

            # compute loss
            loss = self.loss_fn(outputs[0], data[-2], count_loss=False)

            # save results of CMCL (assignment of model)
            if self.config["model"]["version"] != "IE":
                logits = [logit.data.cpu() for logit in logit_list]
                vis_data = [*self.sample_data, self.criterion.assignments]
                if type(vis_data[0][-1]) == type(list()):
                    vis_data[0][-1] = vis_data[0][-1][0]
                if self.use_knowledge_distillation:
                    vis_data.append([net_utils.get_data(bout) for bout in self.base_outs])

                vis_utils.save_mcl_visualization(
                    self.config, vis_data, logits, self.itow, self.itoa, prefix, \
                    self.use_knowledge_distillation, self.use_initial_assignment \
                )

                self.save_assignments(prefix, mode)

    """ Status related methods """
    def reset_status(self):
        if self.status == None:
            # initialize metric scores/losses
            self.status = OrderedDict()
            self.status["loss"] = 0
            if self.config["model"]["use_assignment_model"]:
                self.status["sel-loss"] = 0
            self.status["top1"] = 0
            self.status["oracle"] = 0
            if self.config["model"]["use_assignment_model"]:
                self.status["sel-acc"] = 0
        else:
            for k in self.status.keys():
                self.status[k] = 0

        # initialize prediction/gt list
        self.gt_list = []
        self.all_predictions = []
        self.base_pred_all_list = []
        for m in range(self.config["model"]["num_models"]):
            self.base_pred_all_list.append([])

        # initialize question_ids and assignments
        self.qst_ids_list = []
        if self.config["model"]["version"] != "IE":
            self.assignments_list = []

    def compute_status(self, net_output, gts):
        """ Compute status (scores, etc)
        Args:
            net_output: output of network, e.g. logits; variable
            gts: ground-truth; tensor
        """
        if type(gts) == type(list()):
            gts = gts[0] # we use most frequent answers for vqa

        logit_list = net_output[0]
        self.compute_top1_accuracy(logit_list, gts)
        self.compute_oracle_accuracy(logit_list, gts)
        self.compute_confusion_matrix(logit_list, gts)
        if self.use_assignment_model:
            self.compute_assignment_accuracy(
                logit_list, gts, self.assignment_logits)
            self.compute_gt_assignment_accuracy(
                logit_list, gts, self.criterion.assignments)
            if self.criterion.assignment_loss is not None:
                self.status["sel-loss"] = \
                    net_utils.get_data(self.criterion.assignment_loss)[0]
            else:
                self.status["sel-loss"] = -1
            self.counters["sel-loss"].add(self.status["sel-loss"], 1)

        self.attach_predictions()
        self.attach_assignments()

    def print_status(self, epoch, iteration, prefix="", mode="train", is_main_net=True):
        """ Print status (scores, etc)
        Args:
            epoch: current epoch
            iteration: current iteration
            prefix: identity to distinguish models; if is_main_net, this is not needed
            is_main_net: flag about this network is root
        """
        if self.config["model"]["verbose_all"] and False:
            for m in range(self.num_models):
                self.net_list[m].print_status(epoch, iteration, "M%d"%(m+1), is_main_net=False)

        if mode == "train":
            log = getattr(self.logger["train"], "info")
        else:
            log = getattr(self.logger["train"], "debug")

        if is_main_net:
            # prepare txt to print
            jump = 3
            txt = "epoch {} step {} ".format(epoch, iteration)
            for i, (k,v) in enumerate(self.status.items()):
                if (i+1) % jump == 0:
                    txt += ", {} = {:.3f}".format(k, v)
                    log(txt)
                    txt = ""
                elif (i+1) % jump == 1:
                    txt += "{} = {:.3f}".format(k, v)
                else:
                    txt += ", {} = {:.3f}".format(k, v)
            if (self.config["model"]["version"] != "IE") \
                    and (mode != "eval"):
                sls = self.criterion.assignments
                txt += " gt-assign = {}, ".format(
                    "|".join(str(sls[0][i]) for i in range(sls.size(1))))
                if self.use_assignment_model:
                    txt += "pred-assign= {}, ".format(self.pred_assignments[0])
            txt += " ({}|{}|{})".format(
                "|".join(pred for pred in self.basenet_pred),
                utils.label2string(self.itoa, self.top1_predictions[0]), self.top1_gt)

            # print learning information
            log(txt)

    def _create_counters(self):
        self.counters = OrderedDict()
        self.counters["loss"] = accumulator.Accumulator("loss")
        if self.config["model"]["use_assignment_model"]:
            self.counters["sel-loss"] = accumulator.Accumulator("sel-loss")
        self.counters["top1"] = accumulator.Accumulator("top1")
        self.counters["oracle"] = accumulator.Accumulator("oracle")
        if self.config["model"]["use_assignment_model"]:
            self.counters["sel-acc"] = accumulator.Accumulator("sel-acc")

    def bring_loader_info(self, dataset):
        super(Ensemble, self).bring_loader_info(dataset) # Must call super __init__()

        for m in range(self.num_models):
            self.net_list[m].bring_loader_info(dataset)

    def apply_curriculum_learning(self):
        # use Ensemble loss
        self.config["model"]["version"] = self.config["model"]["new_loss"]
        self.criterion = building_blocks.EnsembleLoss(self.config["model"])
        if self.use_gpu:
            self.criterion.cuda()
        self.logger["train"].info(
            "Switching criterion from IE Loss to {}".format(
                self.config["model"]["version"]))

    """ methods for debugging """
    def compare_assignments(self):
        _, assignments = net_utils.get_data(self.assignment_logits).max(dim=1)
        print(torch.stack([self.criterion.assignments.squeeze(), assignments], dim=1))
        print(torch.sum(torch.eq(self.criterion.assignments.squeeze(), assignments)), " - ", assignments.size(0))

    def save_checkpoint(self, cid):
        """ Save checkpoint of the network.
        Args:
            cid: id of checkpoint
        """
        ckpt_path = os.path.join(self.config["misc"]["result_dir"], \
                "checkpoints", "checkpoint_epoch_{:03d}.pkl")
        model_state_dict = {}
        for m in self.model_list:
            model_state_dict[m] = self[m].state_dict()
        torch.save(model_state_dict, ckpt_path.format(cid))
        self.logger["train"].info("Checkpoint is saved in {}".format(
                ckpt_path.format(cid)))

        if self.config["model"]["save_all_net"]:
            ckpt_path = os.path.join(self.config["misc"]["result_dir"], \
                "checkpoints", "M{}_epoch_{:03d}.pkl")
            for m in range(self.num_models):
                torch.save(self.net_list[m].state_dict(),
                           ckpt_path.format(m, cid))
                self.logger["train"].info("M{} is saved in {}".format(
                    m, ckpt_path.format(m, cid)))

            if self.config["model"]["use_assignment_model"]:
                ckpt_path = os.path.join(self.config["misc"]["result_dir"], \
                "checkpoints", "assignment_epoch_{:03d}.pkl")
                torch.save(self.assignment_model.state_dict(),
                           ckpt_path.format(cid))
                self.logger["train"].info("Assignment model is saved in {}".format(
                        ckpt_path.format(cid)))

    @classmethod
    def model_specific_config_update(cls, config):
        assert type(config) == type(dict()), \
            "Configuration shuold be dictionary"

        m_config = config["model"]
        if m_config["base_model_type"] == "saaa":
            config = SAAA.model_specific_config_update(config)
        elif m_config["base_model_type"] == "san":
            config = SAN.model_specific_config_update(config)
        else:
            raise NotImplementedError("Base model type: {}".m_config["base_model_type"])

        if config["model"]["use_assignment_model"]:
            if m_config["base_model_type"] == "san":
                img_emb_config_keys = [
                    "img_emb2d_inp_dim", "img_emb2d_out_dim",
                    "img_emb2d_dropout_prob", "img_emb2d_apply_l2_norm",
                    "img_emb2d_only_l2_norm"]
                for k in img_emb_config_keys:
                    m_config["assignment_"+k] = m_config[k]

            m_config["assignment_fusion_inp1_dim"] = \
                m_config["qst_emb_dim"]
            m_config["assignment_fusion_inp2_dim"] = \
                m_config["assignment_img_emb2d_out_dim"]
            m_config["assignment_mlp_inp_dim"] = m_config["assignment_fusion_dim"]
            m_config["assignment_mlp_out_dim"] = m_config["num_models"]

        return config


class SAN(VirtualVQANetwork):
    def __init__(self, config):
        super(SAN, self).__init__(config) # Must call super __init__()
        self.classname = "SAN"

        self.use_gpu = utils.get_value_from_dict(config["model"], "use_gpu", True)
        loss_reduce = utils.get_value_from_dict(config["model"], "loss_reduce", True)

        # build layers
        self.img_emb_net = building_blocks.Embedding2D(config["model"], "img")
        self.qst_emb_net = building_blocks.QuestionEmbedding(config["model"])
        self.san = building_blocks.StackedAttention(config["model"])
        self.classifier = building_blocks.MLP(config["model"], "answer")
        self.criterion = nn.CrossEntropyLoss(reduce=loss_reduce)

        # set layer names (all and to be updated)
        self.model_list = ["img_emb_net", "qst_emb_net", "san", "classifier", "criterion"]
        self.models_to_update = ["img_emb_net", "qst_emb_net", "san", "classifier", "criterion"]

        self.config = config

    def forward(self, data):
        """ Forward network
        Args:
            data: list [imgs, qst_labels, qst_lenghts, answer_labels]
        Returns:
            logits: logits of network which is an input of criterion
            others: intermediate values (e.g. attention weights for SAN)
        """
        # forward network
        img_feats = self.img_emb_net(data[0])
        qst_feats = self.qst_emb_net(data[1], data[2])
        san_outputs = self.san(qst_feats, img_feats) # [multimodal feat, attention list]
        self.logits = self.classifier(san_outputs[0]) # criterion input

        others = san_outputs[1]
        return [self.logits, others]

    def save_results(self, data, prefix, mode="train", compute_loss=False):
        """ Get qualitative results (attention weights) and save them
        Args:
            data: list [imgs, qst_labels, qst_lenghts, answer_labels, img_paths]
        """
        # save predictions
        self.save_predictions(prefix)

        if mode == "train":
            # maintain sample data
            self._set_sample_data(data)

            # convert data as Variables
            if self.is_main_net:
                data = self.tensor2variable(data)
            # forward network
            logits, att_weight_list = self.forward(data)
            if compute_loss:
                loss = self.loss_fn(logits, data[-2], count_loss=False)

            # visualize result
            qualitative_results = \
                [[weight.data.cpu() for weight in att_weight_list], \
                 logits.data.cpu()]
            vis_utils.save_san_visualization(
                self.config, self.sample_data, qualitative_results, \
                self.itow, self.itoa, prefix)

    @classmethod
    def model_specific_config_update(cls, config):
        assert type(config) == type(dict()), \
            "Configuration shuold be dictionary"

        m_config = config["model"]
        # for attention layer
        m_config["qst_emb_dim"] = m_config["rnn_hidden_dim"]
        m_config["img_emb_dim"] = m_config["img_emb2d_out_dim"]
        # for classigication layer
        m_config["answer_mlp_inp_dim"] = m_config["img_emb2d_out_dim"]
        m_config["answer_mlp_out_dim"] = m_config["num_labels"]

        return config


class SAAA(VirtualVQANetwork):
    def __init__(self, config):
        super(SAAA, self).__init__(config) # Must call super __init__()

        self.classname = "SAAA"
        self.use_gpu = utils.get_value_from_dict(
            config["model"], "use_gpu", True)
        loss_reduce = utils.get_value_from_dict(
            config["model"], "loss_reduce", True)

        # options for KLD with base model
        self.use_knowledge_distillation = \
            utils.get_value_from_dict(
                config["model"], "learning_knowledge_distillation_loss", False)
        base_model_ckpt_path = \
                utils.get_value_from_dict(
                    config["model"], "base_model_ckpt_path", "None")
        if self.use_knowledge_distillation:
            assert base_model_ckpt_path != "None", \
                "checkpoint path for base model should be given"

        # build layers
        self.qst_emb_net = building_blocks.QuestionEmbedding(config["model"])
        self.saaa = building_blocks.RevisedStackedAttention(config["model"])
        self.classifier = building_blocks.MLP(config["model"], "answer")
        if self.use_knowledge_distillation:
            base_config = copy.deepcopy(config)
            base_config["model"]["use_knowledge_distillation"] = False
            self.base_model = SAAA(base_config)
            self.base_model.load_checkpoint(base_model_ckpt_path)
            if self.use_gpu and torch.cuda.is_available():
                self.base_model.cuda()
            self.base_model.eval()
            self.criterion = building_blocks.KLDLoss(config)
        else:
            self.criterion = nn.CrossEntropyLoss(reduce=loss_reduce)

        # set layer names (all and to be updated)
        self.model_list = ["qst_emb_net", "saaa", "classifier", "criterion"]
        self.models_to_update = ["qst_emb_net", "saaa", "classifier", "criterion"]

        self.config = config

    def forward(self, data):
        """ Forward network
        Args:
            data: list [imgs, qst_labels, qst_lenghts, answer_labels]
        Returns:
            logits: logits of network which is an input of criterion
            others: intermediate values (e.g. attention weights)
        """
        # forward network
        # applying l2-norm for img features
        img_feats = \
            data[0] / (data[0].norm(p=2, dim=1, keepdim=True).expand_as(data[0]))
        qst_feats = self.qst_emb_net(data[1], data[2])
        saaa_outputs = self.saaa(qst_feats, img_feats) # [ctx, att list]
        # concatenate attended feature and question feat
        multimodal_feats = torch.cat((saaa_outputs[0], qst_feats), 1)
        self.logits = self.classifier(multimodal_feats) # criterion input

        others = saaa_outputs[1]
        if self.use_knowledge_distillation:
            print("Update using KLD", end="\r")
            teacher_logits = self.base_model.forward(data)[0]
            criterion_inp = [self.logits, teacher_logits]
        else:
            criterion_inp = self.logits
        return [criterion_inp, others]

    def save_results(self, data, prefix, mode="train", compute_loss=False):
        """ Get qualitative results (attention weights) and save them
        Args:
            data: list [imgs, qst_labels, qst_lenghts, answer_labels, img_paths]
        """
        # save predictions
        self.save_predictions(prefix)

        if mode == "train":
            # maintain sample data
            self._set_sample_data(data)

            # convert data as Variables
            if self.is_main_net:
                data = self.tensor2variable(data)
            # forward network
            logits, att_weights = self.forward(data)
            if compute_loss:
                loss = self.loss_fn(logits, data[-2], count_loss=False)

            # visualize result
            if self.use_knowledge_distillation:
                logits = logits[0]
            num_stacks = att_weights.size(1)
            att_weights = att_weights.data.cpu()
            qualitative_results = [[att_weights[:,i,:,:]
                  for i in range(num_stacks)], logits.data.cpu()]
            vis_utils.save_san_visualization(
                self.config, self.sample_data, qualitative_results,
                self.itow, self.itoa, prefix)

    @classmethod
    def model_specific_config_update(cls, config):
        assert type(config) == type(dict()), \
            "Configuration shuold be dictionary"

        m_config = config["model"]
        # for attention layer
        m_config["qst_emb_dim"] = m_config["rnn_hidden_dim"]
        # for classigication layer
        m_config["answer_mlp_inp_dim"] = m_config["rnn_hidden_dim"] \
                + (m_config["img_emb_dim"] * m_config["num_stacks"])
        m_config["answer_mlp_out_dim"] = m_config["num_labels"]

        return config
