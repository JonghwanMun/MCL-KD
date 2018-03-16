import os
import pdb
import glob
import json
import yaml
import logging
import itertools

import h5py
import visdom
import numpy as np
from PIL import Image
from textwrap import wrap
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy as sp
from scipy.misc import imresize
from scipy.ndimage.filters import convolve, gaussian_filter
from collections import OrderedDict

from src.utils import utils, io_utils

try:
    import seaborn as sns
    sns.set_style("whitegrid", {"axes.grid": False})
except ImportError:
    print("Install seaborn to colorful visualization!")
except:
    print("Unknown error")

FONTSIZE  = 5

""" helper functions for visualization """
def resize_attention(attention, width=448, height=448, interp="bilinear", use_smoothing=False):
	resized_attention = imresize(attention, [height, width], interp=interp) / 255
	if use_smoothing:
		resized_attention = gaussian_filter(resized_attention, 15)
	return resized_attention

def overlay_attention_on_image(img, attention):
    im = np.asarray(img, dtype=np.float)
    resized_attention = resize_attention(attention, width=im.shape[1], height=im.shape[0],
                                         interp="bilinear", use_smoothing=True)
    im = im * resized_attention[:,:,np.newaxis]
    new_image = Image.fromarray(np.array(im, dtype=np.uint8))
    return new_image

def add_text_to_figure(fig, gc, row, col, row_height, col_width, texts, rotation=-45,
                       colortype=None, y_loc=0.5, fontsize=FONTSIZE):
    try:
        if colortype == "sequence":
            color = sns.color_palette("Set1", n_colors=23, desat=.4)
        else:
            color = ["black" for _ in range(col)]
    except:
        color = ["black" for _ in range(col)]

    for i, text in enumerate(texts):
        sub = fig.add_subplot(gc[row:row+row_height, col+i:col+i+col_width])

        if col == 0:  # location of index(first col of each row)
            sub.text(0.5, y_loc, "\n".join(wrap(text, 8)), ha="center", va="center",
                     rotation=rotation, fontsize=fontsize, wrap=True)
        else:
            sub.text(0.5, y_loc, text, ha="center", va="center", fontsize=fontsize,
                     rotation=rotation, wrap=True, color=color[i%23])
        sub.axis("off")

def add_image_to_figure(fig, gc, row, col, row_height, col_width, images,
                        show_colorbar=False, vmin=None, vmax=None):
    for i, image in enumerate(images):
        sub = fig.add_subplot(gc[row:row+row_height, col:col+col_width])
        cax = sub.imshow(image, vmin=vmin, vmax=vmax)
        sub.axis("off")
        if show_colorbar:
            fig.colorbar(cax)

def add_vector_to_figure(fig, gc, row, col, row_height, col_width, vectors,
                         ymin=0, ymax=0.1, colortype=None):
    try:
        if colortype == "sequence":
            color = sns.color_palette("Set1", n_colors=23, desat=.4)
        else:
            color = None
    except:
        color = None

    for i, vector in enumerate(vectors):
        sub = fig.add_subplot(gc[row:row+row_height, col:col+col_width])
        #sub.set_ylim([ymin, ymax])
        cax = sub.bar(np.arange(vector.shape[0]), vector, width=0.8, color=color)
        sub.axes.get_xaxis().set_visible(False)
        sub.tick_params(axis="y", which="major", labelsize=3)

def add_question_row_subplot(fig, gc, question, row, col_width=-1):
    if col_width != -1:
        question_width = col_width*2 + 6
    else:
        question_width = 9
    add_text_to_figure(fig, gc, row, 0, 1, 1, ["question"])
    add_text_to_figure(fig, gc, row, 1, 1, question_width, question, rotation=0, colortype="sequence")

def add_attention_row_subplot(fig, gc, img, atts, num_stacks, row, col_width=2):
    for i in range(num_stacks):
        ith_att_weight = atts[i]
        overlayed_att= overlay_attention_on_image(img, ith_att_weight)
        r_pos = row + (i * col_width)

        add_text_to_figure(fig, gc, r_pos, 0, col_width, 1, ["stack_%d" % (i+1)])
        add_image_to_figure(fig, gc, r_pos, 1, col_width, col_width, [img])
        add_image_to_figure(fig, gc, r_pos, col_width+1, col_width, col_width, [overlayed_att])
        add_vector_to_figure(fig, gc, r_pos, col_width*2+1+1, int(col_width/2),
                             col_width, [ith_att_weight.view(14*14).numpy()])
        add_text_to_figure(fig, gc, r_pos+int(col_width/2), col_width*2+1+1, int(col_width/2), col_width,
                           ["max ({:.6f})".format(ith_att_weight.max()), \
                            "min ({:.6f})".format(ith_att_weight.min())])

def add_answer_row_subplot(fig, gc, answer_logit, gt, itoa, row):
    # add ground truth answer
    add_text_to_figure(fig, gc, row, 0, 1, 1, ["GT: {}".format(gt)])

    if type(answer_logit) is list:
        add_text_to_figure(fig, gc, row, 2, 1, 2, \
                ["Selection: {}".format(" | ".join(str(answer_logit[1][i]+1) \
                for i in range(answer_logit[1].size(0))))], rotation=0)
        """
        if isinstance(answer_logit[1], np.int32):
            add_text_to_figure(fig, gc, row, 2, 1, 2, \
                ["Selection: {}".format(str(answer_logit[1]+1))] ,rotation=0)
        else:
            add_text_to_figure(fig, gc, row, 2, 1, 2, \
                ["Selection: {}".format(" | ".join(str(answer_logit[1][s]+1) \
                for s in range(answer_logit[1].shape[0])))], rotation=0)
        """
        for logit in answer_logit[0]:
            # compute probability of answers
            logit = logit.numpy() # tensor to numpy
            answer_prob = np.exp(logit) / (np.exp(logit).sum())  # + 1e-10)
            top5_predictions = ["{}\n({:.3f})".format(itoa[str(a)], answer_prob[a])
                               for i, a in enumerate(answer_prob.argsort()[::-1][:5])]

            add_text_to_figure(fig, gc, row+1, 0, 1, 1, top5_predictions, y_loc=0.5, colortype="sequence")
            add_vector_to_figure(fig, gc, row+1, 6, 1, 3, [answer_prob])
            row += 1
    else:
        # compute probability of answers
        answer_logit = answer_logit.numpy() # tensor to numpy
        answer_prob = np.exp(answer_logit) / (np.exp(answer_logit).sum())  # + 1e-10)
        top5_predictions = ["{}\n({:.3f})".format(itoa[str(a)], answer_prob[a])
                           for i, a in enumerate(answer_prob.argsort()[::-1][:5])]

        add_text_to_figure(fig, gc, row, 1, 1, 1, top5_predictions, y_loc=0.5, colortype="sequence")

def save_san_visualization(config, data, result, itow, itoa, prefix, figsize=(5,5)):
    """ Save visualization of Stacked Attention Network
    Args:
        config: configuration file including information of save directory
        data: list of [imgs, question_labels, question_lengths, answers, img_paths]
        result:list of [attention weights, logits]; (B, h, w), (B, C)
        itow: dictionary for mapping index to word in questions
        itoa: dictionary for mapping index to word in answers
        prefix: name for directory to save visualization
        figsize: figure size
    """
    img_dir = config["train_loader"]["img_dir"]
    save_dir = os.path.join(config["misc"]["result_dir"], "qualitative", "attention")
    io_utils.check_and_create_dir(save_dir)

    attention_weights = result[0]
    logits = result[1]

    img_paths = data[1]
    data = data[0]
    num_data = len(data[2])
    num_stacks = config["model"]["num_stacks"]
    if type(data[-1]) == type(list()):
        data[-1] = data[-1][0]
    for idx in range(num_data):
        # load image
        img_path = img_paths[idx]
        img = Image.open(os.path.join(img_dir, img_path)).convert("RGB")

        # convert indices of question into words and get gt and logit
        question = utils.label2string(itow, data[1][idx])
        gt = utils.label2string(itoa, data[-1][idx])
        logit = logits[idx]

        # create figure
        fig = plt.figure(figsize=figsize)
        col_width = 2
        row = 2 + num_stacks * col_width
        col = col_width*2 + 6
        gc = gridspec.GridSpec(row,col)

        # plot question
        add_question_row_subplot(fig, gc, [question], 0, col_width)
        # plot attention weights
        cur_att = [attention_weights[ns][idx] for ns in range(num_stacks)]
        add_attention_row_subplot(fig, gc, img, cur_att, num_stacks, 1, col_width)
        # plot answers
        add_answer_row_subplot(fig, gc, logit, gt, itoa, row-1)

        # save figure and close it
        img_filename = utils.get_filename_from_path(img_path)
        img_filename = "{}_{}_{}.png".format(idx, prefix, img_filename)
        plt.savefig(os.path.join(save_dir, img_filename), bbox_inches="tight", dpi=500)
        #plt.savefig(save_dir + "_" + img_filename, bbox_inches="tight", dpi=500)
        plt.close()

def save_mcl_visualization(config, data, result, itow, itoa, prefix, use_base_model=False, \
                            use_precomputed_selection=False, figsize=(5,5)):
    """ Save visualization of CMCL-based model
    Args:
        config: configuration file including information of save directory
        data: list of fourcomponents; [[inputs for network], img_info, selections, base_predictions]
            - inputs for network: list of items [imgs, qst_labels, qst_lengths,
                                    (precomputed_selections), answers]
        result: list of logit for m models; m * (B, C)
        itow: dictionary for mapping index to word in questions
        itoa: dictionary for mapping index to word in answers
        prefix: name for directory to save visualization
        figsize: figure size
    """
    # create save directory
    img_dir = config["train_loader"]["img_dir"]
    save_dir = os.path.join(config["misc"]["result_dir"], \
                            "qualitative", "predictions")
    io_utils.check_and_create_dir(save_dir)

    num_data = len(data[0][2])
    #- inputs for network: list of items [imgs, qst_labels, qst_lengths, answers]
    for idx in range(num_data):
        # load image
        img_path = data[1][idx]
        img = Image.open(os.path.join(img_dir, img_path)).convert("RGB")

        # convert indices of question into words and get gt and logit
        question = utils.label2string(itow, data[0][1][idx])
        gt = utils.label2string(itoa, data[0][-1][idx])

        # create figure
        fig = plt.figure(figsize=figsize)
        if use_base_model:
            gc = gridspec.GridSpec(len(result)+2+len(data[3]), 10)
        else:
            gc = gridspec.GridSpec(len(result)+2, 10)

        # plot question
        add_question_row_subplot(fig, gc, [question], 0)

        # plot answers and predictions
        """data: list of fourcomponents; [[inputs for network], img_info, selections, base_predictions]"""
        logits = [logit[idx] for logit in result]
        if use_base_model:
            for i in range(len(data[3])):
                logits.append(data[3][i][idx])
        selections = data[2][idx]
        add_answer_row_subplot(fig, gc, [logits, selections], gt, itoa, 1)

        # save figure and close it
        img_filename = utils.get_filename_from_path(img_path)
        img_filename = "{}_{}_{}.png".format(idx, prefix, img_filename)
        #plt.savefig(os.path.join(save_dir, img_filename), bbox_inches="tight", dpi=500)
        plt.savefig(os.path.join(save_dir, img_filename), bbox_inches="tight", dpi=500)
        plt.close()

def save_confusion_matrix_visualization(config, cm_list, classes, epoch, prefix, fontsize=2,
                                        normalize=True, cmap=plt.cm.Blues, figsize=(5,5)):
    """
    This function save the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)
    if normalize:
        for ii,cm in enumerate(cm_list):
            cm_list[ii] = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # create save directory
    img_dir = config["train_loader"]["img_dir"]
    save_dir = os.path.join(config["misc"]["result_dir"],
                            "qualitative", "confusion_matrix", prefix)
    io_utils.check_and_create_dir(save_dir)

    # create figure
    fig = plt.figure(figsize=figsize)
    gc = gridspec.GridSpec(1, len(cm_list))

    tick_marks = np.arange(len(classes))
    for ii,cm in enumerate(cm_list):
        sub = fig.add_subplot(gc[0, ii])
        # show confusion matrix with colorbar
        ax = sub.imshow(cm, interpolation='nearest', cmap=cmap)
        # show title, axis (labels)
        sub.set_title("model_{}".format(ii), fontsize=3)
        plt.setp(sub, xticks=tick_marks, xticklabels=classes)
        plt.setp(sub, yticks=tick_marks, yticklabels=classes)
        plt.setp(sub.get_xticklabels(), fontsize=fontsize, rotation=-45)
        plt.setp(sub.get_yticklabels(), fontsize=fontsize)

        if len(classes) <= 100:
            fmt = '.2f' if normalize else 'd'
            max_val = cm.max()
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if cm[i, j] > (max_val * 0.4):
                    sub.text(j, i, format(cm[i, j], fmt),
                             horizontalalignment="center", fontsize=fontsize, rotation=45,
                             color="white" if cm[i, j] > (max_val * 0.8) else "black")

        if ii == 0:
            sub.set_ylabel('True label', fontsize=3)
        sub.set_xlabel('Predicted label', fontsize=3)
        """
        if (ii+1) == len(cm_list):
            divider = make_axes_locatable(sub)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(ax, cax=cax)
            cbar.ax.tick_params(labelsize=2)
        """
    fig.tight_layout()

    # save figure and close it
    plt.savefig(os.path.join(save_dir, "epoch_{:03d}.png".format(epoch)), bbox_inches="tight", dpi=450)
    plt.close()
