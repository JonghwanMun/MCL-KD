import os
import pdb
import time
import yaml
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable

from src.model import building_networks
from src.dataset import clevr_dataset, vqa_dataset
from src.experiment import common_functions as cmf
from src.utils import accumulator, timer, utils, io_utils, net_utils

""" Get parameters """
def _get_argument_params():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--config_path",
        default="src/experiment/options/ensemble.yml", help="Path to config file.")
	parser.add_argument("--model_type",
        default="saaa", help="Model type among [san | saaa | ensemble].")
	parser.add_argument("--dataset",
        default="clevr", help="Dataset to train models [clevr | vqa].")
	parser.add_argument("--output_filename",
        default="ensemble", help="filename for predictions.")
	parser.add_argument("--assignment_path",
                     default="None",
#        default="data/CLEVR_v1.0/preprocess/encoded_qa/vocab_train_raw/"
#                     + "all_questions_use_zero_token_max_qst_len_45/"
#                     + "m3_o1_assignment_train.h5",
                     help="Path to assignment file.")
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
    elif params["model_type"] == "ensemble":
        M = getattr(building_networks, "Ensemble")
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

""" Training the network """
def ensemble(config):

    """ Build data loader """
    dset = dataset.DataSet(config["test_loader"])
    L = data.DataLoader( \
            dset, batch_size=config["test_loader"]["batch_size"], \
            num_workers=config["num_workers"], \
            shuffle=False, collate_fn=dataset.collate_fn)

    """ Load assignments if exists """
    with_assignment = False
    if config["assignment_path"] != "None":
        with_assignment = True
        assignment_file = io_utils.load_hdf5(config["assignment_path"], verbose=False)
        assignments = assignment_file["assignments"][:]
        cnt_mapping = np.zeros((3,3))

    """ Build network """
    nets = []
    net_configs = []
    for i in range(len(config["checkpoint_paths"])):
        net_configs.append(io_utils.load_yaml(config["config_paths"][i]))
        net_configs[i] = M.override_config_from_loader(net_configs[i], dset)
        nets.append(M(net_configs[i]))
        nets[i].bring_loader_info(dset)
        apply_cc_after = utils.get_value_from_dict(
                net_configs[i]["model"], "apply_curriculum_learning_after", -1)
        # load checkpoint if exists
        nets[i].load_checkpoint(config["checkpoint_paths"][i])
        start_epoch = int(utils.get_filename_from_path(
                config["checkpoint_paths"][i]).split("_")[-1])
        # If checkpoint use curriculum learning
        if (apply_cc_after > 0) and (start_epoch >= apply_cc_after):
            nets[i].apply_curriculum_learning()

    # ship network to use gpu
    if config["use_gpu"]:
        for i in range(len(nets)):
            nets[i].gpu_mode()
    for i in range(len(nets)):
        nets[i].eval_mode()

    """ Run training network """
    counters = {}
    counters["ensemble"] = accumulator.Accumulator("ensemble")
    counters["oracle"] = accumulator.Accumulator("oracle")
    iter_per_epoch = dset.get_iter_per_epoch()
    for i in range(len(nets)):
        nets[i].reset_status() # initialize status
        modelname = "M{}".format(i)
        counters[modelname] = accumulator.Accumulator(modelname)

    ii = 0
    itoa = dset.get_itoa()
    predictions = []
    for batch in tqdm(L):
        # Forward networks
        probs = 0
        B = batch[0][0].size(0)
        if type(batch[0][-1]) == type(list()):
            gt = batch[0][-1][0]
        else:
            gt = batch[0][-1]

        if with_assignment:
            correct_mask_list = []

        correct = 0
        for i in range(len(nets)):
            outputs = nets[i].evaluate(batch)
            #prob = outputs[1]
            prob = F.softmax(outputs[1], dim=1) # 2018.3.21 4:50 pm

            # count correct numbers for each model
            val, idx = net_utils.get_data(prob).max(dim=1)
            correct = torch.eq(idx, gt)
            num_correct = torch.sum(correct)
            modelname = "M{}".format(i)
            counters[modelname].add(num_correct, B)

            # add prob of each model
            probs = probs + prob
            if i == 0:
                oracle_correct = correct
            else:
                oracle_correct = oracle_correct + correct

            if with_assignment:
                correct_mask_list.append(correct.clone())

        # top1 accuracy for ensemble
        max_val, max_idx = probs.data.cpu().max(dim=1)
        num_correct = torch.sum(torch.eq(max_idx, gt))
        counters["ensemble"].add(num_correct, B)

        # oracle accuracy for ensemble
        num_oracle_correct = torch.sum(torch.ge(oracle_correct, 1))
        counters["oracle"].add(num_oracle_correct, B)

        # attach predictions
        for i in range(len(batch[1])):
            qid = batch[1][i]
            predictions.append({
                "question_id": qid,
                "answer": utils.label2string(itoa, max_idx[i])
            })

        if with_assignment:
            correct_mask_list = torch.stack(correct_mask_list, 1).numpy()
            for it in range(correct_mask_list.shape[0]):
                cur_assignment = assignments[ii]
                cnt_mapping[cur_assignment] += correct_mask_list[it]
                ii += 1
        # epoch done

    # print accuracy
    for m in range(len(nets)):
        print("M{} top1 accuracy: {:.3f}".format(m, counters["M{}".format(m)].get_average()))
    print("Ensembled top1 accuracy: {:.3f}".format(counters["ensemble"].get_average()))
    print("Ensembled oracle accuracy: {:.3f}".format(counters["oracle"].get_average()))

    save_dir = os.path.join("results", "ensemble_predictions")
    io_utils.check_and_create_dir(save_dir)
    io_utils.write_json(os.path.join(save_dir, config["out"]+".json"), predictions)
    if with_assignment:
        np.save(os.path.join(save_dir, "mapping_count"), cnt_mapping)

def main(params):
    # loading configuration and setting environment
    config = io_utils.load_yaml(params["config_path"])
    config["debug_mode"] = params["debug_mode"]
    config["out"] = params["output_filename"]
    config["assignment_path"] = params["assignment_path"]

    # ensemble networks
    ensemble(config)

if __name__ == "__main__":
    params = _get_argument_params()
    _set_model(params)
    _set_dataset(params)
    main(params)
