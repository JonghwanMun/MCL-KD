import os
import math
import h5py
import json
import numpy as np
from PIL import Image
from textwrap import wrap
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy as sp
from scipy.misc import imresize
from scipy.ndimage.filters import convolve, gaussian_filter

import torch
try:
    import seaborn as sns
    sns.set_style("whitegrid", {"axes.grid": False})
except ImportError:
    print("Install seaborn to colorful visualization!")
except:
    print("Unknown error")

from src.utils import io_utils

FONTSIZE  = 5

""" helper functions for QA plot """
def label2string(itow, label):
    """ Convert labels to string (question, caption, etc)
    Args:
        itwo: dictionry for mapping index to word
        label: indices of label
    """
    if torch.typename(label) == "int":
        return itow[str(label)]
    else:
        txt = ""
        for l in label:
            if l == 0:
                break
            else:
                txt += (itow[str(l)] + " ")
        return txt.strip()

""" helper functions for dictionary """
def get_value_from_dict(dict, key, default_value):
    """ Get value from dictionary (if key don"t exists, use default value)
    Args:
        dict: dictionary
        key: key value
        default_value: default_value
    Returns:
        dict[key] or default_value: value from dictionary or default value
    """
    if default_value == None and (dict == None or dict[key] == None):
        print("ERROR: required key " + key + " was not provided in dictionary")
    if dict == None:
        return default_value

    if key in dict.keys():
        return dict[key]
    else:
        return default_value

""" helper functions for string """
def get_filename_from_path(file_path, delimiter="/"):
    """ Get filename from file path (filename.txt -> filename)
    """
    filename = file_path.split(delimiter)[-1]
    return filename.split(".")[0]

""" helper functions for training model """
def adjust_lr(iter, iter_per_epoch, config):
    """ Exponentially decaying learning rate
    Args:
        iter: current iteration
        iter_per_epoch: iterations per epoch
        config: configuation file
    Returns:
        decay_lr: decayed learning rate
    """
    if config["decay_every_epoch"] == -1:
        decay_lr = config["init_lr"]
    else:
        decay_lr = config["init_lr"] * math.exp(math.log(
            config["decay_factor"]) / iter_per_epoch \
            / config["decay_every_epoch"])**iter
    return decay_lr

