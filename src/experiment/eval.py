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
	parser.add_argument("--loader_config_path", default="src/experiment/options/test_loader_config.yml",
            help="Do evaluation or getting/saving some values")
	parser.add_argument("--mode", default="eval", help="Do evaluation or getting/saving some values")
	parser.add_argument("--exp", type=str, required=True, help="Experiment or configuration name")
	parser.add_argument("--model_type", default="ensemble", help="Model type among [san | ensemble | saaa].")
	parser.add_argument("--dataset", default="clevr", help="dataset to train models [clevr|vqa].")
	parser.add_argument("--num_workers", type=int, default=4, help="The number of workers for data loader.")
	parser.add_argument("--start_epoch", type=int, default=10, help="Start epoch to evaluate.")
	parser.add_argument("--end_epoch", type=int, default=50, help="End epoch to evaluate.")
	parser.add_argument("--epoch_stride", type=int, default=5, help="Stride for jumping epoch.")
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
    dset = dataset.DataSet(loader_config)
    L = data.DataLoader(dset, batch_size=loader_config["batch_size"], \
                        num_workers=params["num_workers"], \
                        shuffle=False, collate_fn=dataset.collate_fn)
    config = M.override_config_from_loader(config, dset)

    if params["mode"] == "eval":

        """ Evaluating networks """
        e0 = params["start_epoch"]
        e1 = params["end_epoch"]
        e_stride = params["epoch_stride"]
        sample_data = dset.get_samples(5)
        for epoch in range(e0, e1+1, e_stride):
            """ Build network """
            net = M(config)
            net.bring_loader_info(dset)
            # ship network to use gpu
            if config["model"]["use_gpu"]:
                net.gpu_mode()

            # load checkpoint
            if not (net.classname == "ENSEMBLE" and config["model"]["version"] == "IE"):
                ckpt_path = os.path.join(exp_path, "checkpoints",
                                         "checkpoint_epoch_{:03d}.pkl".format(epoch))
                assert os.path.exists(ckpt_path), \
                    "Checkpoint does not exists ({})".format(ckpt_path)
                net.load_checkpoint(ckpt_path)

            # If checkpoint is already applied with curriculum learning
            apply_cc_after = utils.get_value_from_dict(
                    config["model"], "apply_curriculum_learning_after", -1)
            if (apply_cc_after > 0) and (epoch >= apply_cc_after):
                net.apply_curriculum_learning()

            cmf.evaluate(config, L, net, epoch-1, logger_name="eval",
                         mode="Evaluation", verbose_every=100)

    elif params["mode"] == "selection":
        epoch = params["start_epoch"]
        """ Build network """
        net = M(config)
        net.bring_loader_info(dset)
        # ship network to use gpu
        if config["model"]["use_gpu"]:
            net.gpu_mode()

        # load checkpoint
        ckpt_path = os.path.join(exp_path, "checkpoints", "checkpoint_epoch_{:03d}.pkl".format(epoch))
        assert os.path.exists(ckpt_path), "Checkpoint does not exists ({})".format(ckpt_path)
        net.load_checkpoint(ckpt_path)
        apply_cc_after = utils.get_value_from_dict(
                config["model"], "apply_curriculum_learning_after", -1)
        # If checkpoint use curriculum learning
        if (apply_cc_after > 0) and (epoch >= apply_cc_after):
            net.apply_curriculum_learning()

        cmf.get_selection_values(config, L, net, epoch-1, logger_name="eval", mode="Evaluation", verbose_every=100)


if __name__ == "__main__":
    params = _get_argument_params()
    global M, dataset
    M = cmf.get_model(params["model_type"])
    dataset = cmf.get_dataset(params["dataset"])
    main(params)
