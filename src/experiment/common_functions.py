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

from src.dataset import clevr_dataset, vqa_dataset
from src.model import building_blocks, building_networks
from src.utils import accumulator, timer, utils, io_utils

""" get base model """
def get_model(base_model_type):
    if base_model_type in ["san", "SAN"]:
        M = getattr(building_networks, "SAN")
    elif base_model_type in ["saaa", "SAAA"]:
        M = getattr(building_networks, "SAAA")
    elif base_model_type in ["sharedsaaa", "SharedSAAA"]:
        M = getattr(building_networks, "SharedSAAA")
    elif base_model_type in ["ensemble", "ENSEMBLE"]:
        M = getattr(building_networks, "Ensemble")
    elif base_model_type in ["infer", "INFER"]:
        M = getattr(building_networks, "EnsembleInference")
    else:
        raise NotImplementedError("Not supported model type ({})".format(base_model_type))
    return M

def get_dataset(dataset):
    if dataset == "clevr":
        D = eval("clevr_dataset")
    elif dataset == "vqa":
        D = eval("vqa_dataset")
    else:
        raise NotImplementedError("Not supported model type ({})".format(dataset))
    return D

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
def evaluate(config, loader, net, epoch, logger_name="epoch", mode="Train", verbose_every=None):

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

        if (config["misc"]["debug"]) and (ii > 5):
            break
        # end for batch in loader

    net.metric = net.counters["top1-avg"].get_average() # would be used for tuning parameter
    net.print_counters_info(epoch+1, logger_name=logger_name, mode=mode)
    net.save_results(None, "epoch_{:03d}".format(epoch+1), mode="eval")

""" validate the network with temperature scaling """
#def evaluate_calibration(config, loader, net, epoch, TT, logger_name="epoch", mode="Train", verbose_every=None):
def evaluate_calibration(config, loader, net, epoch, logger_name="epoch", mode="Train", verbose_every=None):

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

    # initialize counters for different tau
    metrics = ["top1-avg", "top1-max", "oracle"]
    tau = [1.0, 1.2, 1.5, 2.0, 5.0, 10.0, 50.0, 100.0]
    counters = OrderedDict()
    for T in tau:
        tau_name = "tau-"+str(T)
        counters[tau_name] = OrderedDict()
        for mt in metrics:
            counters[tau_name][mt] = accumulator.Accumulator(mt)

    """ Run validating network """
    ii = 0
    tm = timer.Timer()
    for batch in loader:
        data_load_duration = tm.get_duration()
        # forward the network
        tm.reset()
        outputs = net.evaluate(batch)
        run_duration = tm.get_duration()

        B = batch[0][-1].size()[0]
        # accumulate the number of correct answers
        for T in tau:
            #outputs[1][0][:] = [new_logit / T for new_logit in outputs[1][0]]
            new_output = [[new_logit / T for new_logit in outputs[1][0]]]
            net.compute_status(new_output, batch[0][-1])

            tau_name = "tau-"+str(T)
            for mt in metrics:
                counters[tau_name][mt].add(net.status[mt]*B, B)

        # print learning information
        if ((verbose_every > 0) and ((ii+1) % verbose_every == 0)) \
                or config["misc"]["debug"]:
            net.print_status(epoch+1, ii+1, mode="eval")
            txt = "[TEST] fetching for {:.3f}s, inference for {:.3f}s\n"
            logger.debug(txt.format(data_load_duration, run_duration))

        ii += 1
        tm.reset()

        if (config["misc"]["debug"]) and (ii > 5):
            break
        # end for batch in loader

    logger.info("\nAt epoch {}".format(epoch+1))
    for cnt_k,cnt_v in counters.items():
        txt = cnt_k + " "
        for k,v in cnt_v.items():
            txt += ", {} = {:.5f}".format(v.get_name(), v.get_average())
        print(txt)
        logger.info(txt)

    """
    net.metric = net.counters["top1-avg"].get_average() # would be used for tuning parameter
    net.print_counters_info(epoch+1, logger_name=logger_name, mode=mode)
    net.save_results(None, "epoch_{:03d}".format(epoch+1), mode="eval")
    """

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

""" Methods for debugging """
def one_step_forward(L, net):
    # fetch the batch
    batch = next(iter(L))

    # forward and update the network
    # Note that the 1st and 2nd item of outputs from forward should be loss and logits
    # The others would change depending on the network
    outputs = net.forward_update(batch, 0.0005)

    # accumulate the number of correct answers
    net.compute_status(outputs[1], batch[3])

    # print learning status
    net.print_status(epoch, 1)
