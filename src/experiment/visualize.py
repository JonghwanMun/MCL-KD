import os
import time
import yaml
import json
import logging
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable

from src.model import building_networks
from src.dataset import clevr_dataset, vqa_dataset
from src.experiment import common_functions as cmf
from src.utils import accumulator, timer, utils, io_utils, vis_utils

""" Get parameters """
def _get_argument_params():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--exp", type=str, required=True, help="Experiment or configuration name")
	parser.add_argument("--model_type", default="cmcl", help="Model type among [san | cmcl].")
	parser.add_argument("--dataset", default="clevr", help="dataset to train models [clevr|vqa].")
	parser.add_argument("--num_workers", type=int, default=4, help="The number of workers for data loader.")
	parser.add_argument("--start_epoch", type=int, default=10, help="Start epoch to evaluate.")
	parser.add_argument("--end_epoch", type=int, default=50, help="End epoch to evaluate.")
	parser.add_argument("--epoch_stride", type=int, default=5, help="Stride for jumping epoch.")
	parser.add_argument("--interactive" , action="store_true", default=False,
		help="Run the script in an interactive mode")
	parser.add_argument("--debug_mode" , action="store_true", default=False,
		help="Run the script in debug mode")

	params = vars(parser.parse_args())
	print (json.dumps(params, indent=4))
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

def main(params):

    # load configuration of pre-trained models
    exp_path = os.path.join("results", params["dataset"], params["model_type"], params["exp"])
    config_path = os.path.join(exp_path, "config.yml")
    config = io_utils.load_yaml(config_path)
    config["exp_path"] = exp_path

    """ Build data loader """
    dset = {}
    dset = dataset.DataSet(config["train_loader"])
    L = data.DataLoader(dset, batch_size=config["train_loader"]["batch_size"], \
                        num_workers=params["num_workers"], \
                        shuffle=False, collate_fn=dataset.collate_fn)

    """ Evaluating networks """
    e0 = params["start_epoch"]
    e1 = params["end_epoch"]
    e_stride = params["epoch_stride"]
    sample_data = dset.get_samples(10)
    for epoch in range(e0, e1+1, e_stride):
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
                config["training"], "apply_curriculum_learning_after", -1)
        # If checkpoint use curriculum learning
        if (apply_cc_after > 0) and (epoch >= apply_cc_after):
            net.apply_curriculum_learning()

        # visualize outputs for sample data
        prefix = "vis_epoch_{:03d}".format(epoch)
        net._set_sample_data(sample_data)
        # convert data as Variables
        dt = [*net.tensor2variable(sample_data[:-1]), sample_data[-1]]

        outputs = net.forward(dt[:-1])
        logit_list = outputs[0][0]

        # compute loss
        loss = net.loss_fn(outputs[0], dt[-2], count_loss=False)

        # save results of CMCL (selection of model)
        logits = [logit.data.cpu() for logit in logit_list]
        vis_data = [*net.sample_data, net.criterion.selections]
        if net.use_knowledge_distillation:
            #vis_data.append(F.softmax(net.base_outs[0], dim=1).clone().data.cpu())
            vis_data.append(net.base_outs[0].clone().data.cpu())

        vis_utils.save_cmcl_visualization(net.config, vis_data, logits, net.itow, \
                net.itoa, prefix, net.use_knowledge_distillation, \
                net.use_precomputed_selection
        )


if __name__ == "__main__":
    params = _get_argument_params()
    _set_model(params)
    _set_dataset(params)
    if not params["interactive"]:
        main(params)
