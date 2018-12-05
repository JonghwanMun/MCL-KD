import os
import pdb
import copy
import numpy as np
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.model import building_blocks
from src.model.virtual_network import VirtualNetwork
from src.model.virtual_VQA_network import VirtualVQANetwork
from src.experiment import common_functions as cmf
from src.utils import accumulator, utils, io_utils, vis_utils, net_utils

# import externals
import vqa.models as external_models


class Ensemble(VirtualVQANetwork):
    def __init__(self, config):
        super(Ensemble, self).__init__(config) # Must call super __init__()
        self.classname = "ENSEMBLE"

        # options for loading model
        self.base_model_type = utils.get_value_from_dict(
            config["model"], "base_model_type", "san")
        self.M = cmf.get_model(self.base_model_type)
        self.use_gpu = utils.get_value_from_dict(
            config["model"], "use_gpu", True)
        self.num_models = utils.get_value_from_dict(
            config["model"], "num_models", 5)

        # options if use knowledge distillation
        self.use_knowledge_distillation = utils.get_value_from_dict(
                config["model"], "use_knowledge_distillation", False)
        base_model_ckpt_path = utils.get_value_from_dict(
                    config["model"], "base_model_ckpt_path", "None")
        if self.use_knowledge_distillation:
            assert base_model_ckpt_path != "None", \
                "checkpoint path for base model should be given"

        # build and load base models if use
        if self.use_knowledge_distillation:
            self.base_model = []
            num_base_models = len(base_model_ckpt_path)
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
        self.criterion = building_blocks.EnsembleLoss(config["model"])

        # set is_main_net flag of base networks as False
        for m in range(self.num_models):
            self.net_list[m].set_is_main_net(False)

        # set models to update
        self.model_list = ["net_list", "criterion"]
        self.models_to_update = ["net_list", "criterion"]

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

        if self.use_knowledge_distillation:
            self.base_outputs = []
            for i in range(len(self.base_model)):
                base_output = self.base_model[i].forward(data) # [logit, attention]
                self.base_outputs.append(base_output[0])
            criterion_inp[0].append(self.base_outputs)

        return criterion_inp

    """ Visualization related methods """
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

        assignments, qst_ids = cmf.reorder_assignments_using_qst_ids(
                origin_qst_ids, qst_ids, assignments, is_subset=True)

        # setting directory for saving assignments
        save_dir = os.path.join(self.config["misc"]["result_dir"], "assignments", mode)
        io_utils.check_and_create_dir(save_dir)

        # save assignments
        save_hdf5_path = os.path.join(save_dir, prefix + "_assignment.h5")
        hdf5_file = io_utils.open_hdf5(save_hdf5_path, "w")
        hdf5_file.create_dataset("assignments", dtype="int32", data=assignments)
        print("Assignments are saved in {}".format(save_hdf5_path))

        save_json_path = os.path.join(save_dir, prefix + "_assignment.json")
        for qsp in self.assignment_qst_ans_pairs:
            for k,v in qsp.items():
                qsp[k] = list(qsp[k])
        io_utils.write_json(save_json_path, self.assignment_qst_ans_pairs)
        print("Assignments (ans-qst) are saved in {}".format(save_json_path))

        # save assignments of qst_ids
        save_json_path = os.path.join(save_dir, prefix + "_qst_ids.json")
        out = {}
        out["question_ids"] = qst_ids
        io_utils.write_json(save_json_path, out)
        print ("Saving is done: {}".format(save_json_path))

    def visualize_assignments(self, prefix, mode="train"):
        class_names = [self.itoa[str(key)] for key in range(len(self.itoa.keys()))]
        vis_utils.save_assignment_visualization(
                self.config, self.assign_per_model, class_names, prefix, mode)

    def save_results(self, data, prefix, mode="train"):
        """ Save visualization of results (attention weights)
        Args:
            data: list [imgs, qst_labels, qst_lenghts, answer_labels, img_path]
        """

        self.save_predictions(prefix, mode)

        # save assignment
        if self.config["model"]["version"] != "IE":
            self.save_assignments(prefix, mode)
            self.visualize_assignments(prefix=prefix, mode=mode)

        """ given sample data """
        if data is not None:
            # maintain sample data
            self._set_sample_data(data)

            # convert data as Variables
            data = [*self.tensor2variable(data), data[1]]

            outputs = self.forward(data[:-1])
            logit_list = outputs[0][0]

            # compute loss
            loss = self.loss_fn(outputs[0], data[-2], count_loss=False)

            # save results of assignment of model
            if self.config["model"]["version"] != "IE":
                logits = [logit.data.cpu() for logit in logit_list]
                vis_data = [*self.sample_data, self.criterion.assignments]
                if type(vis_data[0][-1]) == type(list()):
                    vis_data[0][-1] = vis_data[0][-1][0]
                if self.use_knowledge_distillation:
                    vis_data.append([net_utils.get_data(bout)
                                     for bout in self.base_outputs])

                class_names = [self.itoa[str(key)]
                               for key in range(len(self.itoa.keys()))]
                vis_utils.save_mcl_visualization(
                    self.config, vis_data, logits, class_names, \
                    self.itow, self.itoa, prefix, \
                    self.use_knowledge_distillation
                )

    """ Status related methods """
    def reset_status(self, init_reset=False):
        super(Ensemble, self).reset_status(init_reset)
        self.probs = None
        self.prob_list = None

        # initialize question_ids and assignments
        self.qst_ids_list = []
        if self.config["model"]["version"] != "IE":
            self.assignments_list = []
            self.assignment_qst_ans_pairs = []
            for mi in range(self.num_models):
                self.assignment_qst_ans_pairs.append(defaultdict(lambda: set()))

        if not init_reset:
            self.assign_per_model = torch.zeros(self.num_models, len(self.itoa))

    def compute_status(self, net_output, gts):
        """ Compute status (scores, etc)
        Args:
            net_output: output of network, e.g. logits; variable
            gts: ground-truth; tensor
        """
        # setting prob_list and probs as None
        self.probs = None
        self.prob_list = None

        logit_list = net_output[0]
        self.compute_top1_accuracy(logit_list, gts)
        self.compute_oracle_accuracy(logit_list, gts)
        self.compute_confusion_matrix(logit_list, gts)

        self.attach_predictions()
        self.attach_assignments(gts)

    def print_status(self, epoch, iteration, prefix="",
                     mode="train", is_main_net=True):
        """ Print status (scores, etc)
        Args:
            epoch: current epoch
            iteration: current iteration
            prefix: identity to distinguish models; if is_main_net, this is not needed
            is_main_net: flag about this network is root
        """
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

            txt += " ({}->{}/{})".format("|".join(pred for pred in self.basenet_pred),
                    utils.label2string(self.itoa, self.top1_predictions[0]), self.top1_gt)

            # print learning information
            log(txt)

    def _create_counters(self):
        self.counters = OrderedDict()
        self.counters["loss"] = accumulator.Accumulator("loss")
        self.counters["top1-avg"] = accumulator.Accumulator("top1-avg")
        self.counters["oracle"] = accumulator.Accumulator("oracle")

    def bring_loader_info(self, dataset):
        super(Ensemble, self).bring_loader_info(dataset)

        for m in range(self.num_models):
            self.net_list[m].bring_loader_info(dataset)

    def apply_curriculum_learning(self):
        # use Ensemble loss
        self.logger["train"].info("Switching criterion from {} to {}".format(
                self.config["model"]["version"], self.config["model"]["new_loss"]))
        self.config["model"]["version"] = self.config["model"]["new_loss"]
        self.criterion = building_blocks.EnsembleLoss(self.config["model"])
        if self.use_gpu:
            self.criterion.cuda()

    @classmethod
    def model_specific_config_update(cls, config):
        assert type(config) == type(dict()), \
            "Configuration shuold be dictionary"

        m_config = config["model"]
        if m_config["base_model_type"] == "san":
            config = SAN.model_specific_config_update(config)
        elif m_config["base_model_type"] == "mlp":
            config = SimpleMLP.model_specific_config_update(config)
        else:
            raise NotImplementedError("Base model type: {}".format(
                    m_config["base_model_type"]))

        return config

class SAN(VirtualVQANetwork):
    def __init__(self, config, verbose=True):
        super(SAN, self).__init__(config, verbose) # Must call super __init__()

        self.classname = "SAN"
        self.use_gpu = utils.get_value_from_dict(
            config["model"], "use_gpu", True)
        loss_reduce = utils.get_value_from_dict(
            config["model"], "loss_reduce", True)

        # options for applying l2-norm
        self.apply_l2_norm = \
            utils.get_value_from_dict(config["model"], "apply_l2_norm", False)

        # build layers
        self.img_emb_net = building_blocks.ResBlock2D(config["model"], "img_emb")
        self.qst_emb_net = building_blocks.QuestionEmbedding(config["model"])
        self.saaa = building_blocks.StackedAttention(config["model"])
        self.classifier = building_blocks.MLP(config["model"], "answer")
        self.criterion = nn.CrossEntropyLoss(reduce=loss_reduce)

        # set layer names (all and to be updated)
        self.model_list = ["img_emb_net", "qst_emb_net", "saaa", "classifier", "criterion"]
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
        img_feats = data[0]
        if self.apply_l2_norm:
            img_feat_norm = img_feats.norm(p=2, dim=1, keepdim=True)
            img_feats = img_feats / img_feat_norm.expand_as(data[0])
        img_feats = self.img_emb_net(img_feats)["feats"]
        qst_feats = self.qst_emb_net(data[1], data[2])
        san_outputs = self.saaa(qst_feats, img_feats) # [ctx, att list]
        # concatenate attended feature and question feat
        multimodal_feats = torch.cat((san_outputs[0], qst_feats), 1)
        self.logits = self.classifier(multimodal_feats) # criterion input

        others = san_outputs[1]
        criterion_inp = self.logits
        out = [criterion_inp, others]
        return out

    def save_results(self, data, prefix, mode="train", compute_loss=False):
        """ Get qualitative results (attention weights) and save them
        Args:
            data: list [imgs, qst_labels, qst_lenghts, answer_labels, img_paths]
        """
        # save predictions
        self.save_predictions(prefix, mode)

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
            num_stacks = att_weights.size(1)
            att_weights = att_weights.data.cpu()
            qualitative_results = [[att_weights[:,i,:,:]
                  for i in range(num_stacks)], logits.data.cpu()]
            vis_utils.save_san_visualization(
                    self.config, self.sample_data, qualitative_results,
                    self.itow, self.itoa, prefix)

    @classmethod
    def model_specific_config_update(cls, config):
        assert type(config) == type(dict()), "Configuration shuold be dictionary"

        m_config = config["model"]
        # for image embedding layer
        m_config["img_emb_dim"] = m_config["img_emb_res_block_2d_out_dim"]
        # for attention layer
        m_config["qst_emb_dim"] = m_config["rnn_hidden_dim"]
        # for classigication layer
        m_config["answer_mlp_inp_dim"] = m_config["rnn_hidden_dim"] \
                + (m_config["img_emb_dim"] * m_config["num_stacks"])
        m_config["answer_mlp_out_dim"] = m_config["num_labels"]

        return config

class SimpleMLP(VirtualVQANetwork):
    def __init__(self, config, verbose=True):
        # Must call super __init__()
        super(SimpleMLP, self).__init__(config, verbose)

        self.classname = "SimpleMLP"
        self.use_gpu = utils.get_value_from_dict(
            config["model"], "use_gpu", True)
        loss_reduce = utils.get_value_from_dict(
            config["model"], "loss_reduce", True)

        # options for applying l2-norm
        self.apply_l2_norm = utils.get_value_from_dict(
                config["model"], "apply_l2_norm", False)

        # build layers
        self.img_emb_net = building_blocks.ResBlock2D(config["model"], "img_emb")
        self.qst_emb_net = building_blocks.QuestionEmbedding(config["model"])
        self.classifier = building_blocks.MLP(config["model"], "answer")
        self.criterion = nn.CrossEntropyLoss(reduce=loss_reduce)

        # set layer names (all and to be updated)
        self.model_list = ["img_emb_net", "qst_emb_net", "classifier", "criterion"]
        self.models_to_update = ["img_emb_net", "qst_emb_net", "classifier", "criterion"]

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
        img_feats = data[0]
        if self.apply_l2_norm:
            img_feat_norm = img_feats.norm(p=2, dim=1, keepdim=True)
            img_feats = img_feats / img_feat_norm.expand_as(data[0])
        img_feats = self.img_emb_net(img_feats)["feats"]
        img_feats = img_feats.mean(2).mean(2)
        qst_feats = self.qst_emb_net(data[1], data[2])
        # concatenate attended features and question features
        multimodal_feats = torch.cat((img_feats, qst_feats), 1)
        self.logits = self.classifier(multimodal_feats) # criterion input

        criterion_inp = self.logits
        out = [criterion_inp]
        return out

    def save_results(self, data, prefix, mode="train", compute_loss=False):
        """ Get qualitative results (attention weights) and save them
        Args:
            data: list [imgs, qst_labels, qst_lenghts, answer_labels, img_paths]
        """
        # save predictions
        self.save_predictions(prefix, mode)

    @classmethod
    def model_specific_config_update(cls, config):
        assert type(config) == type(dict()), "Configuration shuold be dictionary"

        m_config = config["model"]
        # for image embedding layer
        m_config["img_emb_dim"] = m_config["img_emb_res_block_2d_out_dim"]
        # for classification layer
        m_config["answer_mlp_inp_dim"] = \
            m_config["rnn_hidden_dim"] + m_config["img_emb_dim"]
        m_config["answer_mlp_out_dim"] = m_config["num_labels"]

        return config
