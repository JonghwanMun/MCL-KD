import os
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
    parser.add_argument("--loader_config_path",
                        default="src/experiment/options/test_loader_config.yml",\
                        help="Do evaluation or getting/saving some values")
    parser.add_argument("--exp", type=str, required=True,
                        help="Experiment or configuration name")
    parser.add_argument("--epoch", type=int, required=True,
                        help="Epoch to be tested")
    parser.add_argument("--model_type", default="ensemble",
                        help="Model type among [san | ensemble | saaa].")
    parser.add_argument("--dataset", default="vqa",
                        help="dataset to train models [clevr|vqa].")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="The number of workers for data loader.")
    parser.add_argument("--debug_mode" , action="store_true", default=False,
                        help="Run the script in debug mode")

    params = vars(parser.parse_args())
    print (json.dumps(params, indent=4))
    return params

def main(params):

    # load configuration of pre-trained models
    exp_path = os.path.join("results", params["dataset"],
                            params["model_type"], params["exp"])
    config_path = os.path.join(exp_path, "config.yml")
    config = io_utils.load_yaml(config_path)
    params["config_path"] = config_path
    config = M.override_config_from_params(config, params)
    config["exp_path"] = exp_path
    cmf.create_save_dirs(config["misc"])

    # create logger
    logger_path = os.path.join(config["exp_path"], "evaluation.log")
    logger = io_utils.get_logger("Evaluate", log_file_path=logger_path)

    """ Build data loader """
    loader_config = io_utils.load_yaml(params["loader_config_path"])
    dset, L = cmf.get_loader(dataset, ["test"], [loader_config],
                             num_workers=params["num_workers"])
    config = M.override_config_from_loader(config, dset["test"])

    """ Build network """
    ckpt_path = os.path.join(exp_path, "checkpoints",
            "checkpoint_epoch_{:03d}.pkl".format(params["epoch"]))
    net = cmf.factory_model(config, M, dset["test"], ckpt_path)

    """ Test networks """
    cmf.test_inference(config, L["test"], net)


if __name__ == "__main__":
    params = _get_argument_params()
    global M, dataset
    M = cmf.get_model(params["model_type"])
    dataset = cmf.get_dataset(params["dataset"])
    main(params)
