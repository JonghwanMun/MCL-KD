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
            config["model"], "base_model_type", "saaa")
        self.M = cmf.get_model(self.base_model_type)
        self.use_gpu = utils.get_value_from_dict(
            config["model"], "use_gpu", True)
        self.num_models = utils.get_value_from_dict(
            config["model"], "num_models", 5)

        # options if use knowledge distillation
        self.use_knowledge_distillation = utils.get_value_from_dict(
                config["model"], "use_knowledge_distillation", False)
        self.use_initial_assignment = utils.get_value_from_dict(
                config["model"], "use_initial_assignment", False)
        base_model_ckpt_path = utils.get_value_from_dict(
                    config["model"], "base_model_ckpt_path", "None")
        if self.use_knowledge_distillation:
            assert base_model_ckpt_path != "None", \
                "checkpoint path for base model should be given"

        # options if use attention transfer
        self.use_attention_transfer = utils.get_value_from_dict(
                config["model"], "use_attention_transfer", False)

        # options if use shared question embedding layer
        self.use_shared_question_embedding = utils.get_value_from_dict(
            config["model"], "use_shared_question_embedding", False)
        base_qstemb_ckpt_path = utils.get_value_from_dict(
                    config["model"], "base_qstemb_ckpt_path", "None")
        if self.use_knowledge_distillation and self.use_shared_question_embedding:
            assert base_qstemb_ckpt_path != "None", \
                "checkpoint path for base question embedding net should be given"

        # options if use assignment model
        self.use_assignment_model = utils.get_value_from_dict(
            config["model"], "use_assignment_model", False)

        # build base models if use
        if self.use_knowledge_distillation:
            if self.use_shared_question_embedding:
                self.base_qst_emb_net = building_blocks.QuestionEmbedding(config["model"])
                qstemb_state_dict = torch.load(base_qstemb_ckpt_path)
                self.base_qst_emb_net.load_state_dict(qstemb_state_dict)

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
        if self.use_shared_question_embedding:
            self.qst_emb_net = building_blocks.QuestionEmbedding(config["model"])
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
        if self.use_shared_question_embedding:
            self.model_list.append("qst_emb_net")
            self.models_to_update.append("qst_emb_net")
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
        if self.use_shared_question_embedding:
            qst_feats = self.qst_emb_net(data[1], data[2])
        self.logit_list = []
        ce_loss_list = []
        if self.use_attention_transfer:
            net_att_groups = []
            base_net_att_groups = []
        for m in range(self.num_models):
            if self.use_shared_question_embedding:
                ith_net_outputs = self.net_list[m](data, qst_feats)
            else:
                ith_net_outputs = self.net_list[m](data)
            if self.use_attention_transfer:
                net_att_groups.append(ith_net_outputs[2])
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
            if self.use_shared_question_embedding:
                qst_feats = self.base_qst_emb_net(data[1], data[2])
            self.base_outs = []
            for i in range(len(self.base_model)):
                if self.use_shared_question_embedding:
                    base_output = self.base_model[i].forward(data, qst_feats) # [logit, attention]
                    self.base_outs.append(base_output[0])
                else:
                    base_output = self.base_model[i].forward(data) # [logit, attention]
                    self.base_outs.append(base_output[0])
                if self.use_attention_transfer:
                    base_net_att_groups.append(base_output[2])
            criterion_inp[0].append(self.base_outs)

        if self.use_attention_transfer:
            criterion_inp[0].append(net_att_groups)
            criterion_inp[0].append(base_net_att_groups)

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

        """
        # save assignments of qst_ids
        save_json_path = os.path.join(save_dir, prefix + "_qst_ids.json")
        out = {}
        out["question_ids"] = qst_ids
        io_utils.write_json(save_json_path, out)
        print ("Saving is done: {}".format(save_json_path))
        """

    def visualize_assignments(self, prefix, mode="train"):
        class_names = [self.itoa[str(key)] for key in range(len(self.itoa.keys()))]
        vis_utils.save_assignment_visualization(
                self.config, self.assign_per_model, class_names, prefix, mode)

    def save_results(self, data, prefix, mode="train"):
        """ Save visualization of results (attention weights)
        Args:
            data: list [imgs, qst_labels, qst_lenghts, answer_labels, img_path]
        """
        # save predictions
        self.save_predictions(prefix, mode)
        # save visualization of confusion matrix for each model
        if self.config["misc"]["dataset"] != "vqa":
            epoch = int(prefix.split("_")[-1])
            self.visualize_confusion_matrix(epoch, prefix=mode)
            if self.config["model"]["version"] != "IE":
                self.save_assignments(prefix, mode)
                #self.visualize_assignments(prefix=prefix, mode=mode)

        """ given sample data """
        if (data is not None):# and (self.config["misc"]["dataset"] != "vqa"):
            # maintain sample data
            self._set_sample_data(data)

            # convert data as Variables
            data = [*self.tensor2variable(data), data[1]]

            if self.config["model"]["vis_individual_net"]:
                # save visualization of base network
                # for SAN, we visualize attention weights
                logit_list = []
                loss_list = []

                if self.use_shared_question_embedding:
                    qst_feats = self.qst_emb_net(data[1], data[2])
                for m in range(self.num_models):
                    if self.use_shared_question_embedding:
                        self.net_list[m].save_results(data, qst_feats,
                            "{}_m{}-att".format(prefix, m+1), compute_loss=True)
                    else:
                        self.net_list[m].save_results(data,
                            "{}_m{}-att".format(prefix, m+1), compute_loss=True)
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
                    # TODO: implement to use pre-computed logits of base models
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

                class_names = [self.itoa[str(key)] for key in range(len(self.itoa.keys()))]
                vis_utils.save_mcl_visualization(
                    self.config, vis_data, logits, class_names, \
                    self.itow, self.itoa, prefix, \
                    self.use_knowledge_distillation, self.use_initial_assignment \
                )


    """ Status related methods """
    def reset_status(self, init_reset=False):
        if self.status == None:
            # initialize metric scores/losses
            self.status = OrderedDict()
            self.status["loss"] = 0
            if self.config["model"]["use_assignment_model"]:
                self.status["sel-loss"] = 0
            self.status["top1-avg"] = 0
            self.status["top1-max"] = 0
            self.status["oracle"] = 0
            for m in range(self.config["model"]["num_models"]):
                self.status["oracle-{}".format(m+1)] = 0
            if self.config["model"]["use_assignment_model"]:
                self.status["sel-acc"] = 0
        else:
            for k in self.status.keys():
                self.status[k] = 0

        # initialize prediction/gt list
        self.gt_list = []
        self.all_predictions = []
        self.base_pred_all_list = []
        if self.config["misc"]["dataset"] == "vqa":
            self.base_all_predictions = []
        for m in range(self.config["model"]["num_models"]):
            self.base_pred_all_list.append([])
            if self.config["misc"]["dataset"] == "vqa":
                self.base_all_predictions.append([])

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
        if type(gts) == type(list()):
            gts = gts[0] # we use most frequent answers for vqa

        # setting prob_list and probs as None
        self.prob_list = None
        self.probs = None

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
        self.attach_assignments(gts)

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
            txt += " ({}->{}/{})".format(
                "|".join(pred for pred in self.basenet_pred),
                utils.label2string(self.itoa, self.top1_predictions[0]), self.top1_gt)

            # print learning information
            log(txt)

            if self.use_tf_summary and self.training_mode:
                self.write_status_summary(iteration)

    def _create_counters(self):
        self.counters = OrderedDict()
        self.counters["loss"] = accumulator.Accumulator("loss")
        if self.config["model"]["use_assignment_model"]:
            self.counters["sel-loss"] = accumulator.Accumulator("sel-loss")
        self.counters["top1-avg"] = accumulator.Accumulator("top1-avg")
        self.counters["top1-max"] = accumulator.Accumulator("top1-max")
        self.counters["oracle"] = accumulator.Accumulator("oracle")
        for m in range(self.config["model"]["num_models"]):
            metric_name = "oracle-{}".format(m+1)
            self.counters[metric_name] = accumulator.Accumulator(metric_name)
        if self.config["model"]["use_assignment_model"]:
            self.counters["sel-acc"] = accumulator.Accumulator("sel-acc")

    def bring_loader_info(self, dataset):
        super(Ensemble, self).bring_loader_info(dataset)

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
        print(torch.sum(torch.eq(self.criterion.assignments.squeeze(), assignments)), \
              " - ", assignments.size(0))

    def save_checkpoint(self, cid):
        """ Save checkpoint of the network.
        Args:
            cid: id of checkpoint; e.g. epoch
        """
        super(Ensemble, self).save_checkpoint(cid)

        if self.config["model"]["save_all_net"]:
            for m in range(self.num_models):
                self.net_list[m].save_checkpoint(cid, "M{}".format(m))

            if self.use_shared_question_embedding:
                ckpt_path = os.path.join(self.config["misc"]["result_dir"], \
                    "checkpoints", "qstemb_epoch_{:03d}.pkl")
                torch.save(self.qst_emb_net.state_dict(), ckpt_path.format(cid))
                self.logger["train"].info(
                    "Shared question embedding net is saved in {}".format(
                        ckpt_path.format(cid)))

            if self.config["model"]["use_assignment_model"]:
                ckpt_path = os.path.join(self.config["misc"]["result_dir"], \
                    "checkpoints", "assign_model_epoch_{:03d}.pkl")
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
        elif m_config["base_model_type"] == "sharedsaaa":
            config = SharedSAAA.model_specific_config_update(config)
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
        self.save_predictions(prefix, mode)
        if self.config["misc"]["dataset"] != "vqa":
            epoch = int(prefix.split("_")[-1])
            self.visualize_confusion_matrix(epoch, prefix=mode)

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
    def __init__(self, config, verbose=True):
        super(SAAA, self).__init__(config, verbose) # Must call super __init__()

        self.classname = "SAAA"
        self.use_gpu = utils.get_value_from_dict(
            config["model"], "use_gpu", True)
        loss_reduce = utils.get_value_from_dict(
            config["model"], "loss_reduce", True)

        # options for deep image embedding
        self.use_deep_img_emb = utils.get_value_from_dict(
            config["model"], "use_deep_img_embedding", False)
        # options for applying l2-norm
        self.apply_l2_norm = \
            utils.get_value_from_dict(
                config["model"], "apply_l2_norm", True)

        # options if use attention transfer
        self.use_attention_transfer = utils.get_value_from_dict(
                config["model"], "use_attention_transfer", False)
        if self.use_attention_transfer:
            assert self.use_deep_img_emb, \
                "If you use attention transfer, you also use deep image embedding layers"

        # build layers
        if self.use_deep_img_emb:
            self.img_emb_net = building_blocks.ResBlock2D(config["model"], "img_emb")
        self.qst_emb_net = building_blocks.QuestionEmbedding(config["model"])
        self.saaa = building_blocks.RevisedStackedAttention(config["model"])
        self.classifier = building_blocks.MLP(config["model"], "answer")
        self.criterion = nn.CrossEntropyLoss(reduce=loss_reduce)

        # set layer names (all and to be updated)
        self.model_list = ["qst_emb_net", "saaa", "classifier", "criterion"]
        self.models_to_update = ["qst_emb_net", "saaa", "classifier", "criterion"]
        if self.use_deep_img_emb:
            self.model_list.append("img_emb_net")
            self.models_to_update.append("img_emb_net")

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
            img_feats = img_feats \
                / (img_feats.norm(p=2, dim=1, keepdim=True).expand_as(data[0]))
        if self.use_deep_img_emb:
            img_emb_out = self.img_emb_net(img_feats)
            img_feats = img_emb_out["feats"]
            if self.use_attention_transfer:
                att_groups = img_emb_out["att_groups"]
        qst_feats = self.qst_emb_net(data[1], data[2])
        saaa_outputs = self.saaa(qst_feats, img_feats) # [ctx, att list]
        # concatenate attended feature and question feat
        multimodal_feats = torch.cat((saaa_outputs[0], qst_feats), 1)
        self.logits = self.classifier(multimodal_feats) # criterion input

        others = saaa_outputs[1]
        criterion_inp = self.logits
        out = [criterion_inp, others]
        if self.use_attention_transfer:
            out.append(att_groups)
        return out

    def Inference_forward(self, data):
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
            img_feats = img_feats \
                / (img_feats.norm(p=2, dim=1, keepdim=True).expand_as(data[0]))
        if self.use_deep_img_emb:
            img_emb_out = self.img_emb_net(img_feats)
            img_feats = img_emb_out["feats"]
            if self.use_attention_transfer:
                att_groups = img_emb_out["att_groups"]
        qst_feats = self.qst_emb_net(data[1], data[2])
        saaa_outputs = self.saaa(qst_feats, img_feats) # [ctx, att list]
        # concatenate attended feature and question feat
        multimodal_feats = torch.cat((saaa_outputs[0], qst_feats), 1)
        self.logits = self.classifier.Inference_forward(multimodal_feats) # criterion input
        criterion_inp = self.logits
        return criterion_inp

    def save_results(self, data, prefix, mode="train", compute_loss=False):
        """ Get qualitative results (attention weights) and save them
        Args:
            data: list [imgs, qst_labels, qst_lenghts, answer_labels, img_paths]
        """
        # save predictions
        self.save_predictions(prefix, mode)
        if self.config["misc"]["dataset"] != "vqa":
            epoch = int(prefix.split("_")[-1])
            self.visualize_confusion_matrix(epoch, prefix=mode)

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
        assert type(config) == type(dict()), \
            "Configuration shuold be dictionary"

        m_config = config["model"]
        # for image embedding layer
        if "img_emb_res_block_2d_out_dim" in m_config.keys():
            m_config["img_emb_dim"] = m_config["img_emb_res_block_2d_out_dim"]
        # for attention layer
        m_config["qst_emb_dim"] = m_config["rnn_hidden_dim"]
        # for classigication layer
        m_config["answer_mlp_inp_dim"] = m_config["rnn_hidden_dim"] \
                + (m_config["img_emb_dim"] * m_config["num_stacks"])
        m_config["answer_mlp_out_dim"] = m_config["num_labels"]

        return config

class SharedSAAA(VirtualVQANetwork):
    def __init__(self, config, verbose=True):
        super(SharedSAAA, self).__init__(config, verbose) # Must call super __init__()

        self.classname = "SharedSAAA"
        self.use_gpu = utils.get_value_from_dict(
            config["model"], "use_gpu", True)
        loss_reduce = utils.get_value_from_dict(
            config["model"], "loss_reduce", True)

        # options for deep image embedding
        self.use_deep_img_emb = utils.get_value_from_dict(
            config["model"], "use_deep_img_embedding", False)
        # options for applying l2-norm
        self.apply_l2_norm = \
            utils.get_value_from_dict(
                config["model"], "apply_l2_norm", True)
        # options if use attention transfer
        self.use_attention_transfer = utils.get_value_from_dict(
                config["model"], "use_attention_transfer", False)
        if self.use_attention_transfer:
            assert self.use_deep_img_emb, \
                "If you use attention transfer, you also use deep image embedding layers"

        # build layers
        if self.use_deep_img_emb:
            self.img_emb_net = building_blocks.ResBlock2D(config["model"], "img_emb")
        self.saaa = building_blocks.RevisedStackedAttention(config["model"])
        self.classifier = building_blocks.MLP(config["model"], "answer")
        self.criterion = nn.CrossEntropyLoss(reduce=loss_reduce)

        # set layer names (all and to be updated)
        self.model_list = ["saaa", "classifier", "criterion"]
        self.models_to_update = ["saaa", "classifier", "criterion"]
        if self.use_deep_img_emb:
            self.model_list.append("img_emb_net")
            self.models_to_update.append("img_emb_net")

        self.config = config

    def forward(self, data, qst_feats):
        """ Forward network
        Args:
            data: list [imgs, qst_labels, qst_lenghts, answer_labels]
            qst_feats: question embedding features
        Returns:
            logits: logits of network which is an input of criterion
            others: intermediate values (e.g. attention weights)
        """
        # forward network
        # applying l2-norm for img features
        img_feats = data[0]
        if self.apply_l2_norm:
            img_feats = img_feats \
                / (img_feats.norm(p=2, dim=1, keepdim=True).expand_as(data[0]))
        if self.use_deep_img_emb:
            img_emb_out = self.img_emb_net(img_feats)
            img_feats = img_emb_out["feats"]
            if self.use_attention_transfer:
                att_groups = img_emb_out["att_groups"]
        saaa_outputs = self.saaa(qst_feats, img_feats) # [ctx, att list]
        # concatenate attended feature and question feat
        multimodal_feats = torch.cat((saaa_outputs[0], qst_feats), 1)
        self.logits = self.classifier(multimodal_feats) # criterion input

        others = saaa_outputs[1]
        criterion_inp = self.logits
        out = [criterion_inp, others]
        if self.use_attention_transfer:
            out.append(att_groups)
        return out

    def save_results(self, data, qst_feats, prefix, mode="train", compute_loss=False):
        """ Get qualitative results (attention weights) and save them
        Args:
            data: list [imgs, qst_labels, qst_lenghts, answer_labels, img_paths]
        """
        # save predictions
        self.save_predictions(prefix, mode)
        if self.config["misc"]["dataset"] != "vqa":
            epoch = int(prefix.split("_")[-1])
            self.visualize_confusion_matrix(epoch, prefix=mode)

        if mode == "train":
            # maintain sample data
            self._set_sample_data(data)

            # convert data as Variables
            if self.is_main_net:
                data = self.tensor2variable(data)
            # forward network
            outputs = self.forward(data, qst_feats)
            logits, att_weights = outputs[0], outputs[1]
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
        assert type(config) == type(dict()), \
            "Configuration shuold be dictionary"

        m_config = config["model"]
        # for image embedding layer
        if "img_emb_res_block_2d_out_dim" in m_config.keys():
            m_config["img_emb_dim"] = m_config["img_emb_res_block_2d_out_dim"]
        # for attention layer
        m_config["qst_emb_dim"] = m_config["rnn_hidden_dim"]
        # for classigication layer
        m_config["answer_mlp_inp_dim"] = m_config["rnn_hidden_dim"] \
                + (m_config["img_emb_dim"] * m_config["num_stacks"])
        m_config["answer_mlp_out_dim"] = m_config["num_labels"]

        return config

class EnsembleInference(VirtualVQANetwork):
    def __init__(self, config, verbose=True):
        super(EnsembleInference, self).__init__(config, verbose) # Must call super __init__()

        self.classname = "INFERENCE"
        self.use_gpu = utils.get_value_from_dict(
            config["model"], "use_gpu", True)
        loss_reduce = utils.get_value_from_dict(
            config["model"], "loss_reduce", True)
        self.use_knowledge_distillation = False
        self.use_logit = utils.get_value_from_dict(
            config["model"], "use_logit_for_inference", "saaa")

        # options for base model
        self.base_model_type = utils.get_value_from_dict(
            config["model"], "base_model_type", "saaa")
        self.M = cmf.get_model(self.base_model_type)
        base_ckpt_path = utils.get_value_from_dict(
            config["model"], "base_model_ckpt_path", True)

        self.save_sample_mean = utils.get_value_from_dict(
            config["model"], "save_sample_mean", False)
        self.output_with_internal_values = utils.get_value_from_dict(
            config["model"], "output_with_internal_values", False)

        # options for question embedding network
        self.use_qst_emb_net = utils.get_value_from_dict(
            config["model"], "use_qst_emb_net", "saaa")

        # build layers
        self.base_model = []
        self.num_base_models = len(base_ckpt_path)
        for m in range(self.num_base_models):
            base_config = copy.deepcopy(config)
            self.base_model.append(self.M(base_config))
            self.base_model[m].load_checkpoint(base_ckpt_path[m])
            if self.use_gpu and torch.cuda.is_available():
                self.base_model[m].cuda()
            self.base_model[m].eval() # set to eval mode for base model
            self.logger["train"].info( \
                    "{}th base-net is initialized from {}".format( \
                    m, base_ckpt_path[m]))
        if self.use_qst_emb_net:
            self.qst_emb_net = building_blocks.QuestionEmbedding(config["model"])
        self.infer = building_blocks.MLP(config["model"], "ens_infer")
        self.criterion = nn.CrossEntropyLoss(reduce=True)

        # set layer names (all and to be updated)
        self.model_list = ["infer", "criterion"]
        self.models_to_update = ["infer", "criterion"]
        if self.use_qst_emb_net:
            self.model_list.append("qst_emb_net")
            self.models_to_update.append("qst_emb_net")

        if config["flag_inference"]:
            self.mean_vector = torch.load(config["vector_path"] + "sample_mean.pkl")
        self.config = config

    def forward(self, data):
        """ Forward network
        Args:
            data: list [imgs, qst_labels, qst_lenghts, answer_labels]
        Returns:
            logits: logits of network which is an input of criterion
            others: intermediate values (e.g. attention weights)
        """
        B = data[0].size()[0]
        net_probs = []
        mth_net_output = []
        if self.output_with_internal_values:
            for m in range(self.num_base_models): mth_net_output.append([])

        Model_list = ["M0", "M1", "M2", "M3", "M4"]
        for m in range(self.num_base_models):
            if self.save_sample_mean:
                net_outputs = self.base_model[m].Inference_forward(data)
                index_list = [0, 3, 4] # 0 -3072, 1&2&3 - 1024, 4 - 3000
                output = 0
                index_count = 0
                for i in index_list:
                    if index_count == 0:
                        output = net_outputs[i]
                        index_count = 1
                    else:
                        output = torch.cat((output, net_outputs[i]), dim=1)
                    if self.output_with_internal_values:
                        mth_net_output[m].append(net_outputs[i])
                net_probs.append(output)

            elif self.config["flag_inference"]:
                net_outputs = self.base_model[m].Inference_forward(data)
                # 0 - input & dropout, 1 - first L, 2 - dropout, 3 -relue, 4 - infal
                if self.config["inference_case"] == 1:
                    index_list = [3, 4] # 0 -3072, 1&2&3 - 1024, 4 - 3000
                    mean_list = [1, 2]
                elif self.config["inference_case"] == 2:
                    index_list = [0, 4]
                    mean_list = [0, 2]
                elif self.config["inference_case"] == 3:
                    index_list = [4]
                    mean_list = [2]
                output = 0
                index_count = 0
                for i in index_list:
                    mean_vector = torch.from_numpy(self.mean_vector[Model_list[m]][mean_list[index_count]]).float().cuda()
                    #mean_vector = Variable(mean_vector)
                    if index_count == 0:
                        temp_output = 0
                        for j in range(3000):
                            NCM_score = torch.norm(net_outputs[i].data - mean_vector[j].repeat(net_outputs[i].size(0), 1), 2, dim=1)
                            if j == 846 or j == 1702:
                                NCM_score.fill_(0)
                            if j == 0:
                                temp_output = NCM_score.view(-1, 1)
                            else:
                                temp_output = torch.cat((temp_output, NCM_score.view(-1, 1)), dim=1)
                        index_count =1
                        output = Variable(temp_output)
                    else:
                        temp_output = 0
                        for j in range(3000):
                            NCM_score = torch.norm(net_outputs[i].data - mean_vector[j].repeat(net_outputs[i].size(0), 1), 2, dim=1)
                            if j == 846 or j == 1702:
                                NCM_score.fill_(0)
                            if j == 0:
                                temp_output = NCM_score.view(-1, 1)
                            else:
                                temp_output = torch.cat((temp_output, NCM_score.view(-1, 1)), dim=1)
                        output = torch.cat((output, Variable(temp_output)), dim=1)

                net_probs.append(output)
                self.use_qst_emb_net = False
            else:
                net_outputs = self.base_model[m](data)
                if self.use_logit:
                    net_probs.append(net_outputs[0])
                else:
                    net_probs.append(F.softmax(net_outputs[0], dim=1))
        if self.use_qst_emb_net:
            qst_feats = self.qst_emb_net(data[1], data[2])
            net_probs.append(qst_feats)

        concat_probs = torch.cat(net_probs, dim=1)
        self.logits = self.infer(concat_probs) # criterion input

        out = [self.logits]
        if self.output_with_internal_values:
            out.append(mth_net_output)
        return out

    def save_sample_mean_per_class(self,):
        pass

    def save_results(self, data, prefix, mode="train", compute_loss=False):
        """ Get qualitative results (attention weights) and save them
        Args:
            data: list [imgs, qst_labels, qst_lenghts, answer_labels, img_paths]
        """
        # save predictions
        self.save_predictions(prefix, mode)
        if self.config["misc"]["dataset"] != "vqa":
            epoch = int(prefix.split("_")[-1])
            self.visualize_confusion_matrix(epoch, prefix=mode)

    @classmethod
    def model_specific_config_update(cls, config):
        assert type(config) == type(dict()), \
            "Configuration shuold be dictionary"

        if config["model"]["base_model_type"] == "saaa":
            config = SAAA.model_specific_config_update(config)
        elif config["model"]["base_model_type"] == "san":
            config = SAN.model_specific_config_update(config)
        elif config["model"]["base_model_type"] == "sharedsaaa":
            config = SharedSAAA.model_specific_config_update(config)
        else:
            raise NotImplementedError("Base model type: {}".format(
                    m_config["base_model_type"]))

        m_config = config["model"]
        # for inference network
        m_config["ens_infer_mlp_inp_dim"] = \
            m_config["num_labels"] * m_config["num_models"]
        m_config["ens_infer_mlp_out_dim"] = m_config["num_labels"]
        if ("use_qst_emb_net" in m_config.keys()) and m_config["use_qst_emb_net"]:
            m_config["ens_infer_mlp_inp_dim"] = \
                m_config["ens_infer_mlp_inp_dim"] + m_config["rnn_hidden_dim"]
        if config["flag_inference"] or config["model"]["save_sample_mean"]:
            m_config["ens_infer_mlp_inp_dim"] = config["num_input_for_inference"]

        return config

class OnlyQuestion(VirtualVQANetwork):
    def __init__(self, config, verbose=True):
        super(OnlyQuestion, self).__init__(config, verbose) # Must call super __init__()

        self.classname = "ONLYQUESTION"
        self.use_gpu = utils.get_value_from_dict(
            config["model"], "use_gpu", True)
        loss_reduce = utils.get_value_from_dict(
            config["model"], "loss_reduce", True)

        ckpt_path = utils.get_value_from_dict(
            config["model"], "checkpoint_path", True)

        # build layers
        self.qst_emb_net = building_blocks.QuestionEmbedding(config["model"])
        self.classifier = building_blocks.MLP(config["model"], "answer")
        self.criterion = nn.CrossEntropyLoss(reduce=loss_reduce)

        # set layer names (all and to be updated)
        self.model_list = ["qst_emb_net", "classifier", "criterion"]
        self.models_to_update = ["qst_emb_net", "classifier", "criterion"]

        # maintain model configuration
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
        qst_feats = self.qst_emb_net(data[1], data[2])
        # concatenate attended feature and question feat
        self.logits = self.classifier(qst_feats) # criterion input

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
        assert type(config) == type(dict()), \
            "Configuration shuold be dictionary"

        m_config = config["model"]
        # for classigication layer
        m_config["answer_mlp_inp_dim"] = m_config["rnn_hidden_dim"]
        m_config["answer_mlp_out_dim"] = m_config["num_labels"]

        return config


class MutanWrapper(VirtualVQANetwork):
    def __init__(self, config, verbose=True):
        super(MutanWrapper, self).__init__(config, verbose) # Must call super __init__()

        self.classname = "MUTAN"
        self.use_gpu = utils.get_value_from_dict(
            config["model"], "use_gpu", True)
        loss_reduce = utils.get_value_from_dict(
            config["model"], "loss_reduce", True)

        # build layers
        self.mutan = external_models.factory(config["model"],
                                            config["wtoi"], config["atoi"],
                                            cuda=self.use_gpu, data_parallel=False)
        self.criterion = nn.CrossEntropyLoss(reduce=loss_reduce)

        # set layer names (all and to be updated)
        self.model_list = ["mutan", "criterion"]
        self.models_to_update = ["mutan", "criterion"]

    def forward(self, data):
        """ Forward network
        Args:
            data: list [imgs, qst_labels, qst_lenghts, answer_labels]
        Returns:
            logits: logits of network which is an input of criterion
            others: intermediate values (e.g. attention weights)
        """
        # forward network
        input_visual = data[0]
        input_question = data[1]
        self.logits = self.mutan(input_visual, input_question)

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
        return config

    @staticmethod
    def override_config_from_loader(config, loader):
        # model: mutan
        config["wtoi"] = loader.get_wtoi()
        config["atoi"] = loader.get_atoi()

        return config
