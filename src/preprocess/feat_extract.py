from __future__ import print_function, division

import os
import random
import argparse

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.misc import imread, imresize

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.models as tv_models
import torchvision.transforms as trn

from src.model import resnet
from src.dataset import clevr_dataset
from src.utils import utils, io_utils

mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

def _convert_ext_from_img_to_np(path):
    return path.replace(".png", ".npy").replace(".jpg", ".npy")


def _get_feat_path(args, img_path):
    if args.data_type == "h5py":
        feat_path = img_path.split("/")[-1][:-4]
    elif args.data_type == "numpy":
        feat_path = _convert_ext_from_img_to_np(
            os.path.join(args.save_dir, img_path))
    else:
        feat_path = "NONE"
    return feat_path

def _feat_exist(args, feat_path, h5py_file=None):
    if feat_path == "NONE":
        return False
    if (args.data_type == "h5py") and (feat_path in [k for k in h5py_file]):
        return True
    elif (args.data_type == "numpy") and os.path.exists(feat_path):
        return True
    else:
        return False

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--save_dir", default="data/CLEVR_v1.0/feats",
                        help="Directory for saving extracted features.")
    parser.add_argument("--image_dir", default="data/CLEVR_v1.0/images",
                        help="Directory for images.")
    parser.add_argument("--num_batch", default=128, type=int, help="batch size")
    parser.add_argument("--image_size", default=224, type=int,
                        help="Image size. VGG 16-layer network uses 224x224 images as an input.")
    parser.add_argument("--feat_type", default="conv5_3",
                        help="Layer to extract feature. [conv5_3 | conv4]")
    parser.add_argument("--data_type", default="h5py",
                        help="Data type to save features. [h5py | numpy]")
    parser.add_argument("--shuffle" , action="store_true", default=False,
                        help="Shuffle the image list to get features")
    parser.add_argument("--debug_mode", action="store_true", default=False, help="debug mode")

    args = parser.parse_args()
    print("Arguments are as follows:\n", args)

    # create save_dir if not exists
    io_utils.check_and_create_dir(args.save_dir)
    """
    if args.data_type == "numpy":
        io_utils.check_and_create_dir(os.path.join(args.save_dir, "train2014"))
        io_utils.check_and_create_dir(os.path.join(args.save_dir, "test2014"))
        io_utils.check_and_create_dir(os.path.join(args.save_dir, "val2014"))
    """

    # build (or load) network
    M = resnet.resnet101(pretrained=True)   # TODO: remove some layers
    M.cuda()
    cudnn.benchmark = True
    M.eval()

    # get image paths from image directory
    img_paths, dir_paths = io_utils.get_filenames_from_dir(args.image_dir)
    for dr in dir_paths:
        io_utils.check_and_create_dir(dr.replace("/images/", "/feats/"))
    if args.shuffle:
        img_paths = random.sample(img_paths, len(img_paths))
    print("Total number of images: {}".format(len(img_paths)))

    # create h5py file if use h5py as data_type
    if args.data_type == "h5py":
        h5py_file = h5py.File(os.path.join(args.save_dir, "img_feats.h5"), "a")
    else:
        h5py_file = None

	# extract features and save them
    n = 0
    bi = 0
    batch = []
    feat_path_list = []
    for i,img_path in enumerate(tqdm(img_paths)):
        feat_path = _get_feat_path(args, img_path)
        if _feat_exist(args, feat_path, h5py_file):
            if args.debug_mode and ((i+1) % 100 == 0):
                print("[{}] exists".format(feat_path))
            continue

        # load and prprocess img
        try:
            img = imread(args.image_dir + img_path, mode="RGB")
            img = imresize(img, (args.image_size, args.image_size), interp="bicubic")
            img = img.transpose(2, 0, 1)[None]
        except:
            print("[Error] fail to load image from {}".format(args.image_dir + img_path))
            continue

        # save img in batch
        bi = bi+1
        batch.append(img)
        feat_path_list.append(feat_path)

        if (bi == args.num_batch) or ((i+1) == len(img_paths)):
            batch_var = np.concatenate(batch, 0).astype(np.float32)
            batch_var = (batch_var / 255.0 - mean) / std
            batch_var = torch.FloatTensor(batch_var).cuda()
            batch_var = Variable(batch_var, volatile=True)

            if args.feat_type == "conv5_3":
                feats = M.get_conv5_feat(batch_var)
            elif args.feat_type == "conv4":
                feats = M.get_conv4_feat(batch_var)
            else:
                raise NotImplementedError("Not supported feature type ({})".format(args.feat_type))

            # save features
            feats = feats.data.cpu().clone().numpy()
            for b in range(bi):
                if args.data_type == "h5py":
                    h5py_file.create_dataset(feat_path_list[b], dtype='float', data=feats[b])
                elif args.data_type == "numpy":
                    np.save(feat_path_list[b], feats[b])
                else:
                    raise NotImplementedError("Not supported data type ({})".format(args.data_type))
                n += 1
                if args.debug_mode and ((n+1) % 5000 == 0):
                    print("{}th feature is saved in {}".format(i+1, feat_path_list[-1]))
                    if args.debug_mode and ((n+1) % 20000 == 0):
                        print(feats[0].shape)
                        print("max value: ", np.max(feats[0]))
                        print("min value: ", np.min(feats[0]))

            # initialize batch index
            bi = 0
            batch = []
            feat_path_list = []

    # close h5py file if use h5py as data_type
    if args.data_type == "h5py":
        h5py_file.close()


if __name__ == "__main__":
    main()
