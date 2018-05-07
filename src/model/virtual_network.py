import os
import pdb
import logging
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from src.utils import accumulator, timer, utils, io_utils, net_utils
from src.utils.tensorboard_utils import PytorchSummary

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
        self.use_tf_summary = False
        self.it = 0 # it: iteration
        self.update_every = 1
        self.qsts = None

        self._create_counters()
        self._get_loggers()
        self.reset_status(init_reset=True)

        self.tm = timer.Timer() # tm: timer

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
        """
        TODO: consider using dictionary of multiple criterions for multiple losses
        for crit_name,crit in self.criterions.items():
            self.loss[crit_name] = crit(criterion_inp, gt)
            self.status[crit_name] = net_utils.get_data(self.loss[crit_name])[0]
            if count_loss:
                self.counters[crit_name].add(self.status[crit_name], 1)
        """
        return self.loss

    def update(self, loss, lr):
        """ Update the network
        Args:
            loss: loss to train the network
            lr: learning rate
        """

        if self.optimizer == None:
            self.optimizer = torch.optim.Adam(self.get_parameters(), lr=lr)
            self.optimizer.zero_grad() # set gradients as zero before update

        self.it +=1
        loss = loss / self.update_every
        loss.backward()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        if self.it % self.update_every == 0:
            self.optimizer.step()
            self.optimizer.zero_grad() # set gradients as zero before updating the network

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
            self.qsts = net_utils.get_data(data[1])

        # Note that return value is a list of at least two items
        # where the 1st and 2nd items should be loss and inputs for criterion layer
        # (e.g. logits), and remaining items would be intermediate values of network
        # that you want to show or check
        #self.tm.reset()
        outputs = self.forward(data)
        #forward_duration = self.tm.get_duration()
        #self.tm.reset()
        loss = self.loss_fn(outputs[0], data[-1], count_loss=True)
        #loss_duration = self.tm.get_duration()
        #self.tm.reset()
        self.update(loss, lr)
        #update_duration = self.tm.get_duration()
        #txt = "forward {:.4f}s | loss {:.4f}s | update {:.4f}s"
        #print(txt.format(
        #    forward_duration, loss_duration, update_duration), end="\r")
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
            self.qsts = net_utils.get_data(data[1])

        # Note that return value is a list of at least two items
        # where the 1st and 2nd items should be loss and inputs for criterion layer
        # (e.g. logits), and remaining items would be intermediate values of network
        # that you want to show or check
        outputs = self.forward(data)
        loss = self.loss_fn(outputs[0], data[-1], count_loss=True)
        return [loss, *outputs]

    def predict(self, batch):
        """ Compute only network's output
        Args:
            batch: list of two components [inputs for network, image information]
                - inputs for network: should be tensors
        """

        # convert data (tensors) as Variables
        if self.is_main_net:
            self.cur_qids = batch[1]
            data = self.tensor2variable(batch)
            self.qsts = net_utils.get_data(data[1])

        outputs = self.forward(data)
        return [*outputs]

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
        self.logger["train"].info("Checkpoint [{}] is saved in {}".format(
            " | ".join(model_state_dict.keys()), ckpt_path))
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
    def reset_status(self, init_reset=False):
        """ Reset (initialize) metric scores or losses (status).
        """
        if self.status == None:
            self.status = OrderedDict()
            self.status["loss"] = 0
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

            if self.use_tf_summary and self.training_mode:
                self.write_status_summary(iteration)


    """ methods for tensorboard """
    def create_tensorboard_summary(self, tensorboard_dir):
        self.use_tf_summary = True
        self.summary = PytorchSummary(tensorboard_dir)

        self.write_params_summary(epoch=0)

    def write_params_summary(self, epoch):
        if self.models_to_update is None:
            for name, param in self.named_parameters():
                self.summary.add_histogram("model/{}".format(name),
                    net_utils.get_data(param).numpy(), global_step=epoch)
        else:
            for m in self.models_to_update:
                for name, param in self[m].named_parameters():
                    self.summary.add_histogram("model/{}/{}".format(m, name),
                        net_utils.get_data(param).numpy(), global_step=epoch)

    def write_status_summary(self, iteration):
        for k,v in self.status.items():
            self.summary.add_scalar('status/' + k, v, global_step=iteration)

    def write_counter_summary(self, epoch, mode):
        for k,v in self.counters.items():
            self.summary.add_scalar(mode + '/counters/' + v.get_name(),
                               v.get_average(), global_step=epoch)

    """ methods for counters """
    def _create_counters(self):
        self.counters = OrderedDict()
        self.counters["loss"] = accumulator.Accumulator("loss")

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

        if self.use_tf_summary:
            self.write_counter_summary(epoch, mode)

        # reset counters
        self.reset_counters()

    def bring_loader_info(self, loader):
        self.logger["train"].warning(
            "You may need to implement method (bring_loader_info)")
        return

    """ wrapper methods of nn.Modules """
    def get_parameters(self):
        if self.models_to_update is None:
            for name, param in self.named_parameters():
                yield param
        else:
            for m in self.models_to_update:
                for name, param in self[m].named_parameters():
                    yield param

    def cpu_mode(self):
        self.logger["train"].info(
            "Setting cpu() for [{}]".format(" | ".join(self.model_list)))
        self.cpu()

    def gpu_mode(self):
        if torch.cuda.is_available():
            self.logger["train"].info(
                "Setting gpu() for [{}]".format(" | ".join(self.model_list)))
            self.cuda()
        else:
            raise NotImplementedError("Available GPU not exists")
        cudnn.benchmark = True

    def train_mode(self):
        self.train()
        self.training_mode = True
        self.logger["train"].info("Setting train() for [{}]".format(" | ".join(self.model_list)))

    def eval_mode(self):
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
        exp_prefix = utils.get_filename_from_path(params["config_path"], delimiter="options/") \
                if "options" in params["config_path"] \
                else utils.get_filename_from_path(params["config_path"], delimiter="results/")[:-7]
        config["misc"]["exp_prefix"] = exp_prefix
        config["misc"]["result_dir"] = os.path.join( "results", exp_prefix)
        config["misc"]["tensorboard_dir"] = os.path.join("tensorboard", exp_prefix)
        config["misc"]["model_type"] = params["model_type"]

        return config
