import os
import pdb
import time
import yaml
import json
import logging
import argparse
import numpy as np
from datetime import datetime

from sigopt import Connection

import torch
import torch.utils.data as data
from torch.autograd import Variable

from src.model import building_networks
from src.dataset import clevr_dataset, vqa_dataset
from src.experiment import common_functions as cmf
from src.utils import accumulator, timer, utils, io_utils

conn = Connection(client_token="AWBJYRXOWOVVEWHOLOWOPISWYOZUYIBFEKQQZVTSOLVAPQYC")
conn.set_api_url("https://api.sigopt.com")

""" Get parameters """
def _get_argument_params():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--config_path",
        default="src/experiment/options/default.yml", help="Path to config file.")
	parser.add_argument("--model_type",
        default="cmcl", help="Model type among [san | cmcl].")
	parser.add_argument("--dataset",
        default="clevr", help="Dataset to train models [clevr | vqa].")
	parser.add_argument("--num_workers", type=int,
        default=4, help="The number of workers for data loader.")
	parser.add_argument("--multi_gpu" , action="store_true", default=False,
		help="Training models with multiple gpus.")
	parser.add_argument("--interactive" , action="store_true", default=False,
		help="Run the script in an interactive mode.")
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

""" Tunning the parameters of network """
def tune_params(config, assignments, dsets, L):

    # overwrite the suggested parameters
    config["model"]["cmcl_loss"]["tau"] = assignments["tau"]
    config["model"]["cmcl_loss"]["beta"] = assignments["beta"]
    config["training"]["init_lr"] = assignments["lr"]

    apply_cc_after = utils.get_value_from_dict(
            config["training"], "apply_curriculum_learning_after", -1) # load checkpoint if exists

    """ Tune params """
    # Build network
    net = M(config)
    net.bring_loader_info(dsets)
    if len(config["model"]["checkpoint_path"]) > 0:
        net.load_checkpoint(config["model"]["checkpoint_path"])
        start_epoch = int(utils.get_filename_from_path(
            config["model"]["checkpoint_path"]).split("_")[-1])
        # If checkpoint use curriculum learning
        if (apply_cc_after > 0) and (start_epoch >= apply_cc_after):
            net.apply_curriculum_learning()

    # ship network to use gpu
    if config["model"]["use_gpu"]:
        net.gpu_mode()

    """ Run training network """
    ii = 0
    iter_per_epoch = dsets["train"].get_iter_per_epoch()
    net.train_mode() # set network as train mode
    for epoch in range(1):
        net.reset_status() # initialize status
        for batch in L["train"]:

            # Forward and update the network
            # Note that the 1st and 2nd item of outputs from forward() should be
            # loss and logits. The others would change depending on the network
            lr = utils.adjust_lr(ii+1, iter_per_epoch, config["training"])
            outputs = net.forward_update(batch, lr)

            # Compute status for current batch: loss, evaluation scores, etc
            net.compute_status(outputs[1], batch[0][-1])

            # print learning status
            if (ii+1) % config["training"]["print_every"] == 0:
                net.print_status(epoch+1, ii+1)

            ii += 1

            if config["training"]["debug"]:
                if ii % 20 == 0:
                    break
            # epoch done

        # print learning information
        net.print_counters_info(epoch+1, logger_name="epoch", mode="Train")

        # curriculum learning
        if (apply_cc_after >= 0) and ((epoch+1) == apply_cc_after):
            net.apply_curriculum_learning()

    # validate network
    logger["epoch"].info(assignments)
    cmf.eval(config, L["test"], net, epoch, logger_name="epoch", mode="Valid")

    return net.metric


def main(params):
    # loading configuration and setting environment
    config = io_utils.load_yaml(params["config_path"])
    config = M.override_config_from_params(config, params)
    cmf.create_save_dirs(config["training"])

    # create loggers
    global logger
    logger = cmf.create_logger(config)

    """ Build data loader """
    dsets = {}
    dsets["train"] = dataset.DataSet(config["train_loader"])
    dsets["test"] = dataset.DataSet(config["test_loader"])
    L = {}
    L["train"] = data.DataLoader( \
            dsets["train"], batch_size=config["train_loader"]["batch_size"], \
            num_workers=config["training"]["num_workers"], \
            shuffle=False, collate_fn=dataset.collate_fn)
    L["test"] = data.DataLoader( \
            dsets["test"], batch_size=config["test_loader"]["batch_size"], \
            num_workers=config["training"]["num_workers"], \
            shuffle=False, collate_fn=dataset.collate_fn)
    config = M.override_config_from_loader(config, dsets["train"])

    # save configuration for later network reproduction
    save_config_path = os.path.join(config["training"]["result_dir"], "config.yml")
    io_utils.write_yaml(save_config_path, config)

    """ create experiment for sigopt """
    experiment = conn.experiments().create(
        name="KD-MCL Tuning",
        parameters=[
            dict(
                name="beta",
                bounds=dict(
                    min=50,
                    max=500
                ),
                type="int"
            ),
            dict(
                name="tau",
                bounds=dict(
                    min=1.0,
                    max=5.0
                ),
                type="double"
            ),
            dict(
                name="lr",
                bounds=dict(
                    min=0.0005,
                    max=0.00005,
                ),
                type="double"
            )
        ],
        metadata=dict(
            template="pytorch_cnn"
        ),
        observation_budget=299
    )

    """  tune parameters of network """
    for _ in range(experiment.observation_budget):

        suggestion = conn.experiments(experiment.id).suggestions().create()
        assignments = suggestion.assignments
        value = tune_params(config, assignments, dsets, L)

        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=value
        )

    assignments = conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments

    print(assignments)

if __name__ == "__main__":
    params = _get_argument_params()
    _set_model(params)
    _set_dataset(params)
    if not params["interactive"]:
        main(params)
