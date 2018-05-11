import os
import sys
sys.path.append("src/externals/vqa")
import time
import yaml
import json
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
        default="src/experiment/options/default.yml", help="Path to config file.")
	parser.add_argument("--model_type",
        default="ensemble", help="Model type among [san | saaa | ensemble].")
	parser.add_argument("--dataset",
        default="clevr", help="Dataset to train models [clevr | vqa].")
	parser.add_argument("--num_workers", type=int,
        default=4, help="The number of workers for data loader.")
	parser.add_argument("--tensorboard_dir" , type=str, default="./tensorboard",
		help="Directory for tensorboard")
	parser.add_argument("--debug_mode" , action="store_true", default=False,
		help="Train the model in debug mode.")

	params = vars(parser.parse_args())
	print(json.dumps(params, indent=4))
	return params

""" Training the network """
def train(config):

    """ Build data loader """
    dsets = {}
    dsets["train"] = dataset.DataSet(config["train_loader"])
    dsets["test"] = dataset.DataSet(config["test_loader"])
    L = {}
    L["train"] = data.DataLoader( \
            dsets["train"], batch_size=config["train_loader"]["batch_size"], \
            num_workers=config["misc"]["num_workers"], \
            shuffle=True, collate_fn=dataset.collate_fn)
    L["test"] = data.DataLoader( \
            dsets["test"], batch_size=config["test_loader"]["batch_size"], \
            num_workers=config["misc"]["num_workers"], \
            shuffle=True, collate_fn=dataset.collate_fn)
    config = M.override_config_from_loader(config, dsets["train"])

    """ Build network """
    net = M(config)
    net.bring_loader_info(dsets)
    logger["train"].info(str(net))
    apply_cc_after = utils.get_value_from_dict(
        config["model"], "apply_curriculum_learning_after", -1)
    # load checkpoint if exists
    if len(config["model"]["checkpoint_path"]) > 0:
        net.load_checkpoint(config["model"]["checkpoint_path"])
        start_epoch = int(utils.get_filename_from_path(
                config["model"]["checkpoint_path"]).split("_")[-1])
        # If checkpoint use curriculum learning
        if (apply_cc_after > 0) and (start_epoch >= apply_cc_after):
            net.apply_curriculum_learning()
    else:
        start_epoch = 0

    # ship network to use gpu
    if config["model"]["use_gpu"]:
        net.gpu_mode()

    # Prepare tensorboard
    net.create_tensorboard_summary(config["misc"]["tensorboard_dir"])

    """ Run training network """
    ii = 0
    tm = timer.Timer() # tm: timer
    iter_per_epoch = dsets["train"].get_iter_per_epoch()
    for epoch in range(start_epoch, config["optimize"]["num_epoch"]):
        net.train_mode() # set network as train mode
        net.reset_status() # initialize status
        for batch in L["train"]:
            data_load_duration = tm.get_duration()

            # maintain sample data to observe learning status
            if ii == 0:
                sample_data = dsets["train"].get_samples(5)
                """ TODO: get samples from both training/test set
                test_sample_data = dsets["test"].get_samples(5))
                """

            # Forward and update the network
            # Note that the 1st and 2nd item of outputs from forward() should be
            # loss and logits. The others would change depending on the network
            tm.reset()
            lr = utils.adjust_lr(ii+1, iter_per_epoch, config["optimize"])
            outputs = net.forward_update(batch, lr)
            run_duration = tm.get_duration()

            # Compute status for current batch: loss, evaluation scores, etc
            net.compute_status(outputs[1], batch[0][-1])

            # print learning status
            if (ii+1) % config["misc"]["print_every"] == 0:
                net.print_status(epoch+1, ii+1)
                txt = "fetching for {:.3f}s, optimizing for {:.3f}s, lr = {:.5f}"
                logger["train"].debug(txt.format( data_load_duration, run_duration, lr))
                logger["train"].info("\n")

            # visualize results
            if (config["misc"]["vis_every"] > 0) \
                    and ((ii+1) % config["misc"]["vis_every"] == 0):
                if config["misc"]["model_type"] == "ensemble":
                    net.save_results(sample_data, "iteration_{}".format(ii+1), mode="train")

            ii += 1
            tm.reset()

            if config["misc"]["debug"]:
                if ii % 150 == 0:
                    break
            # epoch done

        # save network every epoch
        net.save_checkpoint(epoch+1)

        # visualize results
        net.save_results(sample_data, "epoch_{:03d}".format(epoch+1), mode="train")

        # print status (metric) accumulated over each epoch
        net.print_counters_info(epoch+1, logger_name="epoch", mode="Train")

        # validate network
        if (epoch+1) % config["evaluation"]["every_eval"] == 0:
            cmf.evaluate(config, L["test"], net, epoch, logger_name="epoch", mode="Valid")

        # curriculum learning
        if (apply_cc_after >= 0) and ((epoch+1) == apply_cc_after):
            net.apply_curriculum_learning()

        # reset reference time to compute duration of loading data
        tm.reset()


def main():
    # get parameters from cmd
    params = _get_argument_params()
    global M, dataset
    M = cmf.get_model(params["model_type"])
    dataset = cmf.get_dataset(params["dataset"])

    # loading configuration and setting environment
    config = io_utils.load_yaml(params["config_path"])
    config = M.override_config_from_params(config, params)
    cmf.create_save_dirs(config["misc"])

    # create loggers
    global logger
    logger = cmf.create_logger(config)

    # train network
    train(config)

if __name__ == "__main__":
    main()
