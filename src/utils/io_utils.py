import os
import glob
import json
import yaml
import uuid
import logging, logging.handlers
import subprocess

import h5py
import coloredlogs
import numpy as np
from collections import defaultdict


""" Get Logger with given name
Args:
    name: logger name.
    fmt: log format. (default: %(asctime)s:%(levelname)s:%(name)s:%(message)s)
    level: logging level. (default: logging.DEBUG)
    log_file: path of log file. (default: None)
"""
def get_logger(name, log_file_path=None, fmt="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
               print_lev=logging.DEBUG, write_lev=logging.INFO):
    logger = logging.getLogger(name)
    # Add file handler
    if log_file_path:
        formatter = logging.Formatter(fmt)
        file_handler = logging.handlers.RotatingFileHandler(log_file_path)
        file_handler.setLevel(write_lev)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # Add stream handler
    coloredlogs.install(level=print_lev, logger=logger)
    return logger


""" YAML helpers """
def load_yaml(file_path, verbose=True):
    with open(file_path, "r") as f:
        yml_file = yaml.load(f)
    if verbose:
        print("Load yaml file from {}".format(file_path))
    return yml_file

def write_yaml(file_path, yaml_data, verbose=True):
    with open(file_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    if verbose:
        print("Write yaml file in {}".format(file_path))


""" JSON helpers """
def load_json(file_path, verbose=True):
    if verbose:
        print("Load json file from {}".format(file_path))
    return json.load(open(file_path, "r"))

def write_json(file_path, file_data, verbose=True):
    with open(file_path, "w") as outfile:
        json.dump(file_data, outfile)
    if verbose:
        print("Write json file in {}".format(file_path))


""" HDF5 helpers """
def open_hdf5(file_path, mode="r", verbose=True):
    if verbose:
        print("Open hdf5 file from {}".format(file_path))
    return h5py.File(file_path, mode)

def load_hdf5(file_path, verbose=True):
    if verbose:
        print("Load hdf5 file from {}".format(file_path))
    return h5py.File(file_path, "r")


def load_hdf5_as_numpy_array_dict(file_path, target_group=None):
    f = h5py.File(file_path, "r")
    if target_group is None:
        hdf5_dict = {k: np.array(v) for k, v in f.items()}
    else:
        hdf5_dict = {k: np.array(v) for k, v in f[target_group].items()}
    print("Load hdf5 file: {}".format(file_path))
    return hdf5_dict

def print_hdf5_keys(hdf5_file):
    def printname(name):
        print (name)
    hdf5_file.visit(printname)


""" Directory helpers """
def check_and_create_dir(dir_path):
    if not os.path.isdir(dir_path):
        print("Create directory: {}".format(dir_path))
        os.makedirs(dir_path, exist_ok=True)

def get_filenames_from_dir(search_dir_path, all_path=False):
    """ Get filename list from a directory recursively
    """

    filenames = []
    dirs = []
    for (path, dr, files) in os.walk(search_dir_path):
        dirs.append(path)
        for f in files:
            filenames.append(os.path.join(path if all_path else path.replace(search_dir_path, ""), f))
    return filenames, dirs

