import os
import pdb
import logging
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from src.utils import accumulator, utils, io_utils, net_utils

class VirtualNetwork(nn.Module):
    def __init__(self):
        super(VirtualNetwork, self).__init__() # Must call super __init__()

        self.models_to_update = None
        self.sample_data = None
        self.optimizer = None
        self.training_mode = True
        self.is_main_net = True

        self.counters = None
        self.status = None

        self._create_counters()
        self._get_loggers()
        self.reset_status()

    """ methods for forward/backward """
    def forward(self, data):
        """ Forward network
        Args:
            data: list of two components [inputs for network, image information]
                - inputs for network: should be variables
        Returns:
            criterion_inp: input list for criterion
        """
        raise NotImplementedError("Should override a method (forward)")

    def loss_fn(self, criterion_inp, gt, count_loss=True):
        """ Compute loss
        Args:
            criterion_inp: inputs for criterion which is outputs from forward(); list
            gt: ground truth
            count_loss: flag whether accumulating loss or not (training or inference)
        """
        self.loss = self.criterion(criterion_inp, gt)
        self.status["loss"] = net_utils.get_data(self.loss)[0]
        if count_loss:
            self.counters["loss"].add(self.status["loss"], 1)
        return self.loss

    def update(self, loss, lr):
        """ Update the network
        Args:
            loss: loss to train the network
            lr: learning rate
        """
        if self.optimizer == None:
            self.optimizer = torch.optim.Adam(self.get_parameters(), lr=lr)
        self.optimizer.zero_grad() # set gradients as zero before updating the network
        loss.backward()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.optimizer.step()

    def forward_update(self, batch, lr):
        """ Forward and update the network at the same time
        Args:
            batch: list of two components [inputs for network, image information]
                - inputs for network: should be tensors
            lr: learning rate
        """

        # convert data (tensors) as Variables
        if self.is_main_net:
            self.cur_qids = batch[1]
            data = self.tensor2variable(batch)

        # Note that return value is a list of at least two items
        # where the 1st and 2nd items should be loss and inputs for criterion layer
        # (e.g. logits), and remaining items would be intermediate values of network
        # that you want to show or check
        outputs = self.forward(data)
        # TODO: considering multiple loss usage
        loss = self.loss_fn(outputs[0], data[-1], count_loss=True)
        self.update(loss, lr)
        return [loss, *outputs]

    def evaluate(self, batch):
        """ Compute loss and network's output at once
        Args:
            batch: list of two components [inputs for network, image information]
                - inputs for network: should be tensors
        """

        # convert data (tensors) as Variables
        if self.is_main_net:
            self.cur_qids = batch[1]
            data = self.tensor2variable(batch)

        # Note that return value is a list of at least two items
        # where the 1st and 2nd items should be loss and inputs for criterion layer
        # (e.g. logits), and remaining items would be intermediate values of network
        # that you want to show or check
        outputs = self.forward(data)
        loss = self.loss_fn(outputs[0], data[-1], count_loss=True)
        return [loss, *outputs]

    def tensor2variable(self, tensors):
        """ Convert tensors to variables
        Args:
            tensors: input tensors fetched from data loader
        """
        raise NotImplementedError("Should override this function (tensor2variable)")


    """ methods for checkpoint """
    def load_checkpoint(self, ckpt_path):
        """ Load checkpoint of the network.
        Args:
            ckpt_path: checkpoint file path
        """
        self.logger["train"].info("Checkpoint is loaded from {}".format(ckpt_path))
        model_state_dict = torch.load(ckpt_path)
        for m in model_state_dict.keys():
            if m in self.model_list:
                self[m].load_state_dict(model_state_dict[m])
            else:
                self.logger["train"].info("{} is not in {}".format(
                        m, " | ".join(self.model_list)))

        self.logger["train"].info("[{}] are initialized from checkpoint".format(
                " | ".join(model_state_dict.keys())))

    def save_checkpoint(self, ckpt_path):
        """ Save checkpoint of the network.
        Args:
            ckpt_path: checkpoint file path
        """
        self.logger["train"].info("Checkpoint is saved in {}".format(ckpt_path))
        model_state_dict = {}
        for m in self.model_list:
            model_state_dict[m] = self[m].state_dict()
        torch.save(model_state_dict, ckpt_path)

    def _get_loggers(self):
        """ Create logging variables.
        """
        self.logger = {}
        self.logger["train"] = io_utils.get_logger("Train")
        self.logger["epoch"] = io_utils.get_logger("Epoch")
        self.logger["eval"] = io_utils.get_logger("Evaluate")

    """ method for status (metrics) """
    def reset_status(self):
        """ Reset (initialize) metric scores or losses (status).
        """
        if self.status == None:
            self.status = OrderedDict()
            self.status["loss"] = 0
            self.status["top1"] = 0
        else:
            for k in self.status.keys():
                self.status[k] = 0


    def compute_status(self, logits, gts):
        """ Compute metric scores or losses (status).
            You may need to implement this method.
        Args:
            logits: output logits of network.
            gts: ground-truth
        """
        self.logger["train"].warning("You may need to implement method (compute_status).")
        return

    def print_status(self, epoch, iteration, prefix="", is_main_net=True):
        """ Print current metric scores or losses (status).
            You may need to implement this method.
        Args:
            epoch: current epoch
            iteration: current iteration
            prefix: identity to distinguish models; if is_main_net, this is not needed
            is_main_net: flag about whether this network is root (main)
        """
        if is_main_net:
            # prepare txt to print
            txt = "epoch {} step {}".format(epoch, iteration)
            for k,v in self.status.items():
                txt += ", {} = {:.3f}".format(k, v)

            # print learning information
            self.logger["train"].debug(txt)

    """ methods for counters """
    def _create_counters(self):
        self.counters = OrderedDict()
        self.counters["loss"] = accumulator.Accumulator("loss")
        self.counters["top1"] = accumulator.Accumulator("top1")

    def reset_counters(self):
        for k,v in self.counters.items():
            v.reset()

    def print_counters_info(self, epoch, logger_name="epoch", mode="train"):
        # prepare txt to print
        txt = "[{}] {} epoch".format(mode, epoch)
        for k,v in self.counters.items():
            txt += ", {} = {:.3f}".format(v.get_name(), v.get_average())

        # print learning information at this epoch
        assert logger_name in self.logger.keys(), \
                "{} does not belong to loggers".format(logger_name)
        self.logger[logger_name].info(txt)

        # reset counters
        self.reset_counters()

    def bring_loader_info(self, loader):
        self.logger["train"].warning(
            "You may need to implement method (bring_loader_info)")
        return

    """ wrapper methods of nn.Modules """
    def get_parameters(self):
        """ Wrapper for parameters() """
        if self.models_to_update is None:
            for name, param in self.named_parameters():
                yield param
        else:
            for m in self.models_to_update:
                for name, param in self[m].named_parameters():
                    yield param

    def cpu_mode(self):
        """ Wrapper for cpu() """
        self.logger["train"].info(
            "Setting cpu() for [{}]".format(" | ".join(self.model_list)))
        self.cpu()

    def gpu_mode(self):
        """ Wrapper for cuda() """
        if torch.cuda.is_available():
            self.logger["train"].info(
                "Setting gpu() for [{}]".format(" | ".join(self.model_list)))
            self.cuda()
        else:
            raise NotImplementedError("Available GPU not exists")
        cudnn.benchmark = True

    def train_mode(self):
        """ Wrapper for train() """
        self.train()
        self.training_mode = True
        self.logger["train"].info("Setting train() for [{}]".format(" | ".join(self.model_list)))

    def eval_mode(self):
        """ Wrapper for eval() """
        self.eval()
        self.training_mode = False
        self.logger["train"].info("Setting eval() for [{}]".format(" | ".join(self.model_list)))

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    @staticmethod
    def override_config_from_params(config, params):
        config["misc"]["debug"] = params["debug_mode"]
        config["misc"]["dataset"] = params["dataset"]
        config["misc"]["num_workers"] = params["num_workers"]
        config["misc"]["exp_prefix"] = utils.get_filename_from_path(
            params["config_path"])
        config["misc"]["result_dir"] = os.path.join("results",
            utils.get_filename_from_path(params["config_path"],
            delimiter="options/"))
        config["misc"]["model_type"] = params["model_type"]

        return config
