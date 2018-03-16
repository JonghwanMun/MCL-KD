import os
import time
import yaml
import json
import h5py
import logging
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.utils.data as data
from torch.autograd import Variable

from src.model import building_networks
from src.dataset import clevr_dataset, vqa_dataset
from src.experiment import common_functions as cmf
from src.utils import accumulator, timer, utils, io_utils

""" Get parameters """
def _get_argument_params():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--config_path",
        default="./src/experiment/options/cmcl/default.yml", help="Path to config file.")
	parser.add_argument("--checkpoint_path",
        default="None", help="Path to checkpoint.")
	parser.add_argument("--model_type",
        default="cmcl", help="Model type among [san | cmcl].")
	parser.add_argument("--dataset",
        default="clevr", help="Dataset to train models [clevr | vqa].")
	parser.add_argument("--mode",
        default="train", help="dataset type [train | val].")
	parser.add_argument("--num_workers", type=int,
        default=4, help="The number of workers for data loader.")
	parser.add_argument("--debug_mode" , action="store_true", default=False,
		help="Train the model in debug mode.")

	params = vars(parser.parse_args())
	print(json.dumps(params, indent=4))
	return params


""" Set model """
def _set_model(params):
    global M
    if params["model_type"] == "san":
        M = getattr(building_networks, "SAN")
    elif params["model_type"] == "cmcl":
        M = getattr(building_networks, "CMCL")
    elif params["model_type"] == "saaa":
        M = getattr(building_networks, "SAAA")
    else:
        raise NotImplementedError("Not supported model type ({})".format(params["model_type"]))

""" Set dataset """
def _set_dataset(params):
    global dataset
    if params["dataset"] == "clevr":
        dataset = eval("clevr_dataset")
    elif params["dataset"] == "vqa":
        dataset = eval("vqa_dataset")
    else:
        raise NotImplementedError("Not supported dataset ({})".format(params["dataset"]))

if __name__ == "__main__":
    # load parameters
    params = _get_argument_params()
    _set_model(params)
    _set_dataset(params)

    # loading configuration and setting environment
    config = io_utils.load_yaml(params["config_path"])
    config = M.override_config_from_params(config, params)
    cmf.create_save_dirs(config["misc"])

    """ Build data loader """
    if params["mode"] == "train":
        dset = dataset.DataSet(config["train_loader"])
    else:
        dset = dataset.DataSet(config["test_loader"])

    L = data.DataLoader(dset, batch_size=config["train_loader"]["batch_size"], \
                                 num_workers=config["misc"]["num_workers"], \
                                 shuffle=False, collate_fn=dataset.collate_fn)
    config = M.override_config_from_loader(config, dset)

    """ Build network """
    net = M(config)
    net.bring_loader_info(dset)
    if "apply_curriculum_learning_after" in config["model"].keys():
        apply_cc_after = config["model"]["apply_curriculum_learning_after"]
    else:
        apply_cc_after = -1
    # load checkpoint if exists
    if params["checkpoint_path"] != "None":
        net.load_checkpoint(params["checkpoint_path"])
        epoch = int(utils.get_filename_from_path(params["checkpoint_path"]).split("_")[-1])
        # If checkpoint use curriculum learning
        if (apply_cc_after > 0) and (epoch >= apply_cc_after):
            net.apply_curriculum_learning()
    else:
        if len(config["model"]["checkpoint_path"]) > 0:
            net.load_checkpoint(config["model"]["checkpoint_path"])
            epoch = int(utils.get_filename_from_path(
                config["model"]["checkpoint_path"]).split("_")[-1])
            # If checkpoint use curriculum learning
            if (apply_cc_after > 0) and (epoch >= apply_cc_after):
                net.apply_curriculum_learning()
        else:
            epoch = 1

    # ship network to use gpu
    if config["model"]["use_gpu"]:
        net.gpu_mode()

    """ Run training network """
    net.eval_mode() # set network as train mode
    net.reset_status() # initialize status

    # get assignments
    cmf.save_assignments(config, L, net, dset.get_qst_ids(), \
        prefix="CMCL-saaa-beta0^1", mode=params["mode"])
        #prefix="KD-MCL-beta100", mode=params["mode"])

    print("=====> Do Interactive Mode")

    """
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

    # visualize results
    net.visualize_results(sample_data, "epoch_{}".format(epoch+1))

    # print learning information
    net.print_counters_info(epoch+1, logger_name="epoch", mode="Train")

    # validate network
    if (epoch+1) % config["evaluation"]["every_eval"] == 0:
        cmf.eval(config, L["test"], net, epoch, logger_name="epoch", mode="Valid")

    # curriculum learning
    if (apply_cc_after >= 0) and ((epoch+1) == apply_cc_after):
        net.apply_curriculum_learning()
    """
