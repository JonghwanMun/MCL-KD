import os
import pdb
import time
import h5py
import json
import yaml
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict

import torch
import torch.utils.data as data
import torch.nn.functional as F

from src.dataset import clevr_dataset, vqa_dataset
from src.model import building_blocks, building_networks
from src.utils import accumulator, timer, utils, io_utils, net_utils

""" get base model """
def get_model(base_model_type):
    if base_model_type in ["san", "SAN"]:
        M = getattr(building_networks, "SAN")
    elif base_model_type in ["mlp", "MLP"]:
        M = getattr(building_networks, "SimpleMLP")
    elif base_model_type in ["ensemble", "ENSEMBLE"]:
        M = getattr(building_networks, "Ensemble")
    else:
        raise NotImplementedError("Not supported model type ({})".format(base_model_type))
    return M

def get_dataset(dataset):
    if dataset == "clevr":
        D = eval("clevr_dataset")
    else:
        raise NotImplementedError("Not supported model type ({})".format(dataset))
    return D

def get_loader(dataset, loader_name=[], loader_configs=[], num_workers=2):
    assert len(loader_name) > 0
    dsets, L = {}, {}
    for li,ln in enumerate(loader_name):
        dsets[ln] = dataset.DataSet(loader_configs[li])
        L[ln] = data.DataLoader( \
                dsets[ln], batch_size=loader_configs[li]["batch_size"], \
                num_workers=num_workers,
                shuffle=False, collate_fn=dataset.collate_fn)
    return dsets, L

def factory_model(config, M, dset, ckpt_path):
    net = M(config)
    net.bring_loader_info(dset)
    # ship network to use gpu
    if config["model"]["use_gpu"]:
        net.gpu_mode()

    # load checkpoint
    if len(ckpt_path) > 0:
        if not (net.classname == "ENSEMBLE" and config["model"]["version"] == "IE"):
            assert os.path.exists(ckpt_path), \
                "Checkpoint does not exists ({})".format(ckpt_path)
            net.load_checkpoint(ckpt_path)

    # If checkpoint is already applied with curriculum learning
    apply_cc_after = utils.get_value_from_dict(
            config["model"], "apply_curriculum_learning_after", -1)
    if (apply_cc_after > 0) and (epoch >= apply_cc_after):
        net.apply_curriculum_learning()
    return net

def create_save_dirs(config):
    """ Create neccessary directories for training and evaluating models
    """
	# create directory for checkpoints
    io_utils.check_and_create_dir(os.path.join(config["result_dir"], "checkpoints"))
	# create directory for qualitative results
    io_utils.check_and_create_dir(os.path.join(config["result_dir"], "predictions"))
    io_utils.check_and_create_dir(os.path.join(config["result_dir"], "qualitative"))


""" Get logger """
def create_logger(config, train_mode=True):
    if train_mode:
        logger = {}
        logger_path = os.path.join(config["misc"]["result_dir"], "train.log")
        logger["train"] = io_utils.get_logger(
            "Train", log_file_path=logger_path,\
            print_lev=getattr(logging, config["logging"]["print_level"]),\
            write_lev=getattr(logging, config["logging"]["write_level"]))
        epoch_logger_path = os.path.join(
            config["misc"]["result_dir"], "performance.log")
        logger["epoch"] = io_utils.get_logger(
            "Epoch", log_file_path=epoch_logger_path,\
            print_lev=getattr(logging, config["logging"]["print_level"]),\
            write_lev=getattr(logging, config["logging"]["write_level"]))
    else:
        epoch_logger_path = os.path.join(
            config["misc"]["result_dir"], "performance.log")
        logger["epoch"] = io_utils.get_logger(
            "Epoch", log_file_path=epoch_logger_path,\
            print_lev=getattr(logging, config["logging"]["print_level"]),\
            write_lev=getattr(logging, config["logging"]["write_level"]))
    return logger

""" validate the network """
def evaluate(config, loader, net, epoch, logger_name="epoch",
             mode="Train", verbose_every=None):

    if verbose_every == None:
        verbose_every = config["evaluation"]["print_every"]
    # load logger
    if logger_name == "epoch":
        logger = io_utils.get_logger("Train")
    elif logger_name == "eval":
        logger = io_utils.get_logger("Evaluate")
    else:
        raise NotImplementedError()

    net.eval_mode() # set network as evalmode
    net.reset_status() # reset status

    """ Run validating network """
    ii = 0
    tm = timer.Timer()
    for batch in loader:
        data_load_duration = tm.get_duration()
        # forward the network
        tm.reset()
        outputs = net.evaluate(batch)
        run_duration = tm.get_duration()

        # accumulate the number of correct answers
        net.compute_status(outputs[1], batch[0][-1])

        # print learning information
        if ((verbose_every > 0) and ((ii+1) % verbose_every == 0)) \
                or config["misc"]["debug"]:
            net.print_status(epoch+1, ii+1, mode="eval")
            txt = "[TEST] fetching for {:.3f}s, inference for {:.3f}s\n"
            logger.debug(txt.format(data_load_duration, run_duration))

        ii += 1
        tm.reset()

        if (config["misc"]["debug"]) and (ii > 2):
            break
        # end for batch in loader

    net.metric = net.counters["top1-avg"].get_average() # would be used for tuning parameter
    net.print_counters_info(epoch+1, logger_name=logger_name, mode=mode)
    net.save_results(None, "epoch_{:03d}".format(epoch+1), mode="eval")


""" get assignment values for data """
def reorder_assignments_using_qst_ids(origin_qst_ids, qst_ids, assignments, is_subset=False):
    # reordering assignments along with original data order
    print("Reorder assignments along with question id")
    ordered_assignments = []
    ordered_qst_ids = []
    mappings = {qid:i for i,qid in enumerate(qst_ids)}
    for qid in tqdm(origin_qst_ids):
        if is_subset and (qid not in mappings.keys()):
            continue
        idx = mappings[qid]
        ordered_assignments.append(assignments[idx])
        ordered_qst_ids.append(qst_ids[idx])

    print("check order is correct")
    if not is_subset:
        for qid1, qid2 in tqdm(zip(origin_qst_ids, ordered_qst_ids)):
            if qid1 !=  qid2:
                print(qid1, " != ", qid2)
                raise ValueError("The qid order should be same")

    return ordered_assignments, ordered_qst_ids

def get_assignment_values(loader, net, origin_qst_ids):

    net.eval_mode() # set network as evalmode
    net.reset_status() # reset status

    """ Forward network """
    ii = 0
    qst_id_list = []
    assignment_list = []
    for batch in tqdm(loader):
        # forward the network
        outputs = net.evaluate(batch)

        # stacking assignments
        assignment_list.append(net.criterion.assignments.numpy())
        qst_id_list.append(batch[1])

        ii += 1
        # end for batch in loader

    # stacking assignments and qst_id
    assignments = np.vstack(assignment_list)
    qst_ids = []
    for qid in qst_id_list:
        qst_ids.extend(qid)

    return reorder_assignments_using_qst_ids(origin_qst_ids, qst_ids, assignments)

def save_assignments(config, L, net, qst_ids, prefix="", mode="train"):
    assignment_list, qst_ids = \
            get_assignment_values(L, net, qst_ids)
    assignments = np.vstack(assignment_list)
    print("shape of assignments: ", assignments.shape)

    # save assignments
    save_dir = config["test_loader"]["encoded_hdf5_path"].split("/qa_")[0]
    save_hdf5_path = os.path.join(save_dir, prefix + "_assignment_{}.h5".format(mode))
    hdf5_file = h5py.File(save_hdf5_path, "w")
    hdf5_file.create_dataset("assignments", dtype="int32", data=assignments)
    print("Selctions are saved in {}".format(save_hdf5_path))

    # save assignments of qst_ids
    save_json_path = os.path.join(save_dir, prefix + "_qst_ids_{}.json".format(mode))
    out = {}
    out["question_ids"] = qst_ids
    json.dump(out, open(save_json_path, "w"))
    print ("Saving is done: {}".format(save_json_path))

def save_logits(config, L, net, prefix="", mode="train"):

    # save assignments
    save_dir = os.path.join(config["misc"]["result_dir"], "logits", str(prefix))
    io_utils.check_and_create_dir(save_dir)

    for batch in tqdm(L):
        # forward the network
        outputs = net.evaluate(batch)

        if type(outputs[1]) == type(list()):
            logits = net_utils.get_data(torch.stack(outputs[1], 0)) # [m,B,num_answers]
        else:
            logits = net_utils.get_data(outputs[1]) # [B,num_answers]

        # save logits as filename of qid
        for qi,qst_id in enumerate(batch[1]):
            save_path = os.path.join(save_dir, "{}.npy".format(qst_id))
            np.save(save_path, logits[qi].numpy())

def compute_sample_mean_per_class(config, L, net, prefix=""):

    assert net.classname == "INFERENCE", \
        "Currently only suppert ensemble inference network"
    assert config["model"]["save_sample_mean"]
    assert config["model"]["output_with_internal_values"]

    # prepare directory for saving sample mean
    save_dir = os.path.join("results", "sample_mean", str(prefix))
    io_utils.check_and_create_dir(save_dir)

    # construct sample mean variable where the structure is
    # {
    #   "M0": [sample_mean_for_first_layer, sample_mean_for_second_layer, ...,
    #   sample_mean_for_last_layer]
    #   "M1": [sample_mean_for_first_layer, sample_mean_for_second_layer, ...,
    #   sample_mean_for_last_layer]
    #    ...
    #   "Mm": [sample_mean_for_first_layer, sample_mean_for_second_layer, ...,
    #   sample_mean_for_last_layer]
    # }
    # note: size of each sample_mean is [num_answers, feat_dims]
    num_models = net.num_base_models
    num_answers = config["model"]["num_labels"]
    sample_means = {"M{}".format(m):[] for m in range(num_models)}
    sample_cnt = np.zeros((num_answers))

    i = 1
    for batch in tqdm(L):
        # forward the network
        B = batch[0][0].size(0)
        outputs = net.evaluate(batch)
        gt_answers = batch[0][-1]
        if type(gt_answers) == type(list()):
            gt_answers = gt_answers[1] # all answers

        assert len(outputs) > 0
        internal_values = outputs[2] # m * [internal_1, internal_2, ..., internal_n]

        for b in range(B):
            gt_idx = gt_answers[b]
            gt_idx = np.unique(gt_idx.numpy())
            for idx in gt_idx:
                sample_cnt[idx] += 1
            for m in range(num_models):
                modelname = "M{}".format(m)
                num_internal = len(internal_values[m])
                for iv in range(num_internal):
                    if i == 1:
                        sample_means[modelname].append(
                            np.zeros((num_answers, internal_values[m][iv].size(1)))
                        )
                    for idx in gt_idx:
                        sample_means[modelname][iv][idx] += \
                            net_utils.get_data(internal_values[m][iv][b]).numpy()
            i += 1

    # compute mean (divide by counts)
    for modelname in sample_means.keys():
        for i,s in enumerate(sample_means[modelname]):
            # TODO: deal with nan
            sample_means[modelname][i] = s / sample_cnt[:, np.newaxis]

    # save sample mean & count
    save_path = os.path.join(save_dir, "sample_mean.pkl")
    torch.save(sample_means, save_path)
    print("save done in {}".format(save_path))
    save_path = os.path.join(save_dir, "sample_cnt.pkl")
    torch.save(sample_cnt, save_path)
    print("save done in {}".format(save_path))



""" Methods for debugging """
def one_step_forward(L, net):
    # fetch the batch
    batch = next(iter(L))

    # forward and update the network
    # Note that the 1st and 2nd item of outputs from forward should be loss and logits
    # The others would change depending on the network
    outputs = net.forward_update(batch, 0.0005)

    # accumulate the number of correct answers
    net.compute_status(outputs[1], batch[0][-1])

    # print learning status
    net.print_status(1, 1)
