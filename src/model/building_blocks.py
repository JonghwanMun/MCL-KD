import pdb
import numpy as np
from random import shuffle
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.utils import utils, io_utils, net_utils
from src.model.virtual_network import VirtualNetwork as VN

"""
Blocks for layers
e.g., conv1d, conv2d, linear, mlp, etc
"""
def get_conv1d(in_dim, out_dim, k_size, stride=1, padding=0, bias=True,
               dropout=0.0, nonlinear="ReLU", use_batchnorm=False):
    layers = []
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(in_dim))
    layers.append(nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=k_size, \
                            stride=stride, padding=padding, bias=bias))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(out_dim))
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    if nonlinear != "None":
        layers.append(getattr(nn, nonlinear)())
    return nn.Sequential(*layers)

def get_conv2d(in_dim, out_dim, k_size, stride=1, padding=0, bias=True,
               dropout=0.0, nonlinear="ReLU", use_batchnorm=False):
    layers = []
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(in_dim))
    layers.append(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=k_size, \
                            stride=stride, padding=padding, bias=bias))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(out_dim))
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    if nonlinear != "None":
        layers.append(getattr(nn, nonlinear)())
    return nn.Sequential(*layers)

def get_linear(in_dim, out_dim, bias=True, dropout=0.0, nonlinear="ReLU", use_batchnorm=False):
    layers = []
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(in_dim))
    layers.append(nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(out_dim))
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if nonlinear != "None":
        layers.append(getattr(nn, nonlinear)())
    return nn.Sequential(*layers)

def get_mlp(in_dim, out_dim, hidden_dims, bias=True, dropout=0.0, nonlinear="ReLU", use_batchnorm=False):
    layers = []
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(in_dim))

    D = in_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(in_features=D, out_features=dim, bias=bias))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(dim))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        if nonlinear != "None":
            layers.append(getattr(nn, nonlinear)())
        D = dim
    layers.append(nn.Linear(D, out_dim))
    return nn.Sequential(*layers)

def get_mlp2d(in_dim, out_dim, hidden_dims, bias=True, dropout=0.0,
              nonlinear="ReLU", use_batchnorm=False):
    layers = []
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(in_dim))

    D = in_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(in_features=D, out_features=dim, bias=bias))
        layers.append(nn.Conv2d(in_channels=D, out_channels=dim, kernel_size=k_size, \
                            stride=stride, padding=padding, bias=bias))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(dim))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        if nonlinear != "None":
            layers.append(getattr(nn, nonlinear)())
        D = dim
    layers.append(nn.Conv2d(D, out_dim, k_size, stride, padding, bias=bias))
    return nn.Sequential(*layers)

def get_res_block_2d(in_dim, out_dim, hidden_dim):
    layers = []
    # 1st conv
    layers.append(nn.Conv2d(in_dim, hidden_dim, 1, 1, bias=False))
    layers.append(nn.BatchNorm2d(hidden_dim))
    layers.append(nn.ReLU(inplace=True))
    # 2nd conv
    layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False))
    layers.append(nn.BatchNorm2d(hidden_dim))
    layers.append(nn.ReLU(inplace=True))
    # 3rd conv
    layers.append(nn.Conv2d(hidden_dim, out_dim, 1, 1, bias=False))
    layers.append(nn.BatchNorm2d(out_dim))

    return nn.Sequential(*layers)

"""
Layers for networks
"""
class MLP(nn.Module):
    def __init__(self, config, name=""):
        super(MLP, self).__init__() # Must call super __init__()
        if name != "":
            name = name + "_"

        # get configuration
        inp_dim = utils.get_value_from_dict(
            config, name+"mlp_inp_dim", 256)
        out_dim = utils.get_value_from_dict(
            config, name+"mlp_out_dim", 52)
        dropout_prob = utils.get_value_from_dict(
            config, name+"mlp_dropout_prob", 0)
        hidden_dim = utils.get_value_from_dict(
            config, name+"mlp_hidden_dim", (1024,))
        use_batchnorm = utils.get_value_from_dict(
            config, name+"mlp_use_batchnorm", False)
        nonlinear = utils.get_value_from_dict(
            config, name+"mlp_nonlinear_fn", "ReLU")

        # set layers
        self.mlp_1d = get_mlp(inp_dim, out_dim, hidden_dim, \
                dropout=dropout_prob, nonlinear=nonlinear, use_batchnorm=use_batchnorm)

    def forward(self, inp):
        """
        Args:
            inp: [B, inp_dim]
        Returns:
            answer_label : [B, out_dim]
        """
        return self.mlp_1d(inp)

class ResBlock2D(nn.Module):
    def __init__(self, config, name=""):
        super(ResBlock2D, self).__init__() # Must call super __init__()
        if name != "":
            name = name + "_"

        # get configuration
        inp_dim = utils.get_value_from_dict(
            config, name+"res_block_2d_inp_dim", 1024)
        out_dim = utils.get_value_from_dict(
            config, name+"res_block_2d_out_dim", 1024)
        hidden_dim = utils.get_value_from_dict(
            config, name+"res_block_2d_hidden_dim", 512)
        self.num_blocks = utils.get_value_from_dict(
            config, name+"num_blocks", 1)

        # set layers
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            self.blocks.append(get_res_block_2d(inp_dim, out_dim, hidden_dim))

    def forward(self, inp):
        """
        Args:
            inp: [B, inp_dim, H, w]
        Returns:
            answer_label : [B, out_dim, H, w]
        """
        for i in range(self.num_blocks):
            out = self.blocks[i](inp)
            out += inp
            out = F.relu(out)
        return out

class Embedding2D(nn.Module):
    def __init__(self, config, name=""):
        super(Embedding2D, self).__init__() # Must call super __init__()
        if name != "":
            name = name + "_"

        inp_dim = utils.get_value_from_dict(
            config, name+"emb2d_inp_dim", 1024)
        out_dim = utils.get_value_from_dict(
            config, name+"emb2d_out_dim", 256)
        dropout_prob = utils.get_value_from_dict(
            config, name+"emb2d_dropout_prob", 0.0)
        nonlinear = utils.get_value_from_dict(
            config, name+"emb2d_nonlinear_fn", "None")
        batchnorm = utils.get_value_from_dict(
            config, name+"emb2d_use_batchnorm", False)
        self.apply_l2_norm = \
            utils.get_value_from_dict(config, name+"emb2d_apply_l2_norm", False)
        self.only_l2_norm = utils.get_value_from_dict(
            config, name+"emb2d_only_l2_norm", False)

        assert not ((self.apply_l2_norm == False) and (self.only_l2_norm == True)), \
            "You set only_l2_norm as True, but also set apply_l2_norm as False"

        # define layers
        if not self.only_l2_norm:
            self.embedding_2d = get_conv2d(inp_dim, out_dim, 1, 1,
                dropout=dropout_prob, nonlinear=nonlinear, use_batchnorm=batchnorm)

    def forward(self, inp):
        """
        Args:
            inp: [batch_size, inp_dim, h, w]
        Returns:
            x: [batch_size, out_dim, h, w]
        """
        if self.apply_l2_norm:
            inp_size = inp.size()
            inp = inp.transpose(1,2).transpose(2,3)
            inp = inp.resize(inp_size[0], inp_size[2]*inp_size[3], inp_size[1])
            inp = F.normalize(inp, p=2, dim=1)
            inp = inp.resize(inp_size[0], inp_size[2], inp_size[3], inp_size[1])
            inp = inp.transpose(3,2).transpose(2,1)
            if self.only_l2_norm:
                return inp

        out = self.embedding_2d(inp)
        return out

class MultimodalFusion(nn.Module):
    def __init__(self, config, name=""):
        super(MultimodalFusion, self).__init__() # Must call super __init__()
        if name != "":
            name = name + "_"

        # get model options
        self.method = utils.get_value_from_dict(config, name+"fusion_method", "hadamard")

        # build layers
        if self.method == "hadamard":
            # get model options
            inp1_dim = utils.get_value_from_dict(config, name+"fusion_inp1_dim", 256)
            inp1_dropout_prob = utils.get_value_from_dict(
                config, name+"fusion_inp1_dropout_prob", 0)
            inp1_emb_nonlinear_fn = utils.get_value_from_dict(
                config, name+"fusion_inp1_nonlinear_fn", "Tanh")
            inp2_dim = utils.get_value_from_dict(config, name+"fusion_inp2_dim", 256)
            inp2_dropout_prob = utils.get_value_from_dict(
                config, name+"fusion_inp2_dropout_prob", 0)
            inp2_emb_nonlinear_fn = utils.get_value_from_dict(
                config, name+"fusion_inp2_nonlinear_fn", "Tanh")
            multimodal_dim = utils.get_value_from_dict(config, name+"fusion_dim", 256)

            # build layers for hadamard
            self.inp1_emb_layer = get_linear(
                    inp1_dim, multimodal_dim, dropout=inp1_dropout_prob,
                    nonlinear=inp1_emb_nonlinear_fn)
            self.inp2_emb_layer = get_conv2d(
                    inp2_dim, multimodal_dim, k_size=1, bias=False,
                    dropout=inp2_dropout_prob, nonlinear=inp2_emb_nonlinear_fn)
            self.multimodal_emb_layer = get_linear(
                    multimodal_dim, multimodal_dim, nonlinear="None")


    def forward(self, data):
        """ Fuse multimodal features (e.g. hadamard product or element-wise multiplication)
            Currently inp1 and inp2 should be questions (1D) and images (1D or 2D)
        Args:
            inp1: [B, inp1_dim]; question
            inp2: [B, inp2_dim, h, w] or [B, inp2]; image
        Returns:
            multimodal_feat : [B, multimodal_dim]
        """
        inp1 = data[0]
        inp2 = data[1]
        if self.method == "hadamard":
            inp1_emb_feat = self.inp1_emb_layer(inp1)
            inp2_emb_feat = self.inp2_emb_layer(inp2).mean(2).mean(2) # applying avg-pooling
            multimodal_feat = inp1_emb_feat * inp2_emb_feat
            multimodal_feat = self.multimodal_emb_layer(multimodal_feat)
        elif self.method == "elem-mul":
            if inp2.dim() == 4:
                inp2 = inp2.mean(2).mean(2) # applying avg-pooling
            multimodal_feat = inp1 * inp2

        return multimodal_feat


class AssignmentModel(nn.Module):
    def __init__(self, config):
        super(AssignmentModel, self).__init__() # Must call super __init__()
        name = "assignment"

        # build layers
        self.img_emb_layer = Embedding2D(config, name="assignment_img")
        self.qst_emb_layer = QuestionEmbedding(config)
        layers = []
        layers.append(MultimodalFusion(config, name))
        layers.append(MLP(config, name)) # TODO: setting num_models
        self.assign_layer = nn.Sequential(*layers)

    def forward(self, data):
        # fetch data
        imgs = data[0]
        qst_labels = data[1]
        qst_len = data[2]

        img_feats = self.img_emb_layer(imgs)
        qst_feats = self.qst_emb_layer(qst_labels, qst_len)
        assignment_logits = self.assign_layer([qst_feats, img_feats])
        return assignment_logits


class QuestionEmbedding(nn.Module):
    def __init__(self, config, name=""):
        super(QuestionEmbedding, self).__init__() # Must call super __init__()
        if name != "":
            name = name + "_"

        # get configurations
        self.use_gpu = utils.get_value_from_dict(config, "use_gpu", True)

        # options for word embedding
        word_emb_dim = utils.get_value_from_dict(
            config, name+"word_emb_dim", 300)
        padding_idx = utils.get_value_from_dict(
            config, name+"word_emb_padding_idx", 0)
        self.apply_nonlinear = utils.get_value_from_dict(
            config, name+"apply_word_emb_nonlinear", False)
        self.word_emb_dropout_prob = utils.get_value_from_dict(
            config, name+"word_emb_dropout_prob", 0.0)

        # options for rnn
        self.rnn_type = utils.get_value_from_dict(
            config, name+"rnn_type", "LSTM")
        self.num_layers = utils.get_value_from_dict(
            config, name+"rnn_num_layers", 2)
        self.rnn_dim = utils.get_value_from_dict(
            config, name+"rnn_hidden_dim", 256)
        rnn_dropout_prob = utils.get_value_from_dict(
            config, name+"rnn_dropout_prob", 0)
        self.bidirectional = utils.get_value_from_dict(
            config, name+"bidirectional", False)
        vocab_size = utils.get_value_from_dict(config, name+"vocab_size", 10)

        assert (self.rnn_type == "GRU") or (self.rnn_type == "LSTM"),\
                "Not supported RNN type: {}" \
                + "(neither GRU or LSTM)".format(self.rnn_type)

        # word embedding layers
        self.lookuptable = nn.Embedding(
            vocab_size, word_emb_dim, padding_idx=padding_idx)
        if self.word_emb_dropout_prob > 0:
            self.word_emb_dropout = nn.Dropout(self.word_emb_dropout_prob)

        # RNN layers
        self.rnn = getattr(nn, self.rnn_type)(
                input_size=word_emb_dim, hidden_size=self.rnn_dim,
                num_layers=self.num_layers, bias=True,
                batch_first=True, dropout=rnn_dropout_prob,
                bidirectional=self.bidirectional)

    def forward(self, qsts, qst_len):
        """
        Args:
            qsts: [B, t], FloatTensor
            qst_len: [B], list
        Returns:
            ht: last hidden states [B, H] or all hidden states [B, t, H] at the last layer
        """
        B = len(qst_len)

        # forward word embedding
        if self.apply_nonlinear:
            word_emb = F.tanh(self.lookuptable(qsts)) # [B, t, word_emb_dim])
        else:
            word_emb = self.lookuptable(qsts) # [B, t, word_emb_dim])
        if self.word_emb_dropout_prob > 0:
            word_emb = self.word_emb_dropout(word_emb)

        # h: [B, W, hidden_size]
        # forward RNN
        word_emb_packed = pack_padded_sequence(word_emb, qst_len, batch_first=True)
        packed_out, ht = self.rnn(word_emb_packed) # [], [num_layers, B, rnn_dim]
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # return forwarded output
        if self.bidirectional:
            return out # [B, max_len, 2*rnn_dim]
        else:
            idx = torch.LongTensor(np.asarray(qst_len)).type_as(qsts.data).long()
            idx = Variable(idx-1, requires_grad=False)

            idx = idx.view(B, 1, 1).expand(B, 1, out.size(2))
            H = out.size(2)
            return out.gather(1, idx).view(B, H)

    def get_initial_hidden_state(self, batch_size):
        """ Get initial hidden states h0 (and cell states c0 for LSTM)
        Args:
            batch_size: batch_size of input sequence
        """
        init_h = Variable(torch.zeros(self.num_layers, batch_size, self.rnn_dim))
        if self.rnn_type == "LSTM":
            init_c = Variable(torch.zeros(self.num_layers, batch_size, self.rnn_dim))

        if self.use_gpu:
            return init_h.cuda() if self.rnn_type == "GRU" \
                                else (init_h.cuda(), init_c.cuda())
        else:
            return init_h if self.rnn_type == "GRU" else (init_h, init_c)


class RevisedStackedAttention(nn.Module):
    def __init__(self, config):
        super(RevisedStackedAttention, self).__init__() # Must call super __init__()

        # get SAN configurations
        self.num_stacks = utils.get_value_from_dict(config, "num_stacks", 2)
        self.qst_feat_dim = utils.get_value_from_dict(config, "qst_emb_dim", 256)
        img_feat_dim = utils.get_value_from_dict(config, "img_emb_dim", 256)
        self.att_emb_dim = utils.get_value_from_dict(config, "att_emb_dim", 512)
        att_dropout_prob = utils.get_value_from_dict(config, "att_dropout_prob", 0.5)

        assert self.num_stacks > 0, "# of stacks {} < 1.".format(self.num_stacks)

        # build layers
        layers = []
        if att_dropout_prob > 0:
            layers.append(nn.Dropout(p=att_dropout_prob))
        layers.append(nn.Conv2d(img_feat_dim+self.qst_feat_dim, self.att_emb_dim, 1, 1))
        layers.append(nn.ReLU())
        if att_dropout_prob > 0:
            layers.append(nn.Dropout(p=att_dropout_prob))
        layers.append(nn.Conv2d(self.att_emb_dim, self.num_stacks, 1, 1))
        self.att_encoder = nn.Sequential(*layers)
        self.att_softmax = nn.Softmax(dim=2)

    def forward(self, qst_feat, img_feat):
        """ Compute context vector given qst feature and visual features
        Args:
            qst_feat: [B, qst_feat_dim]
            img_feat: [B, img_feat_dim, h, w]
        Returns:
            ctx_feat: [B, ctx_feat_dim]
            att_weights_list: attention weithgs [B, s, h, w]
        """

        B, K, H, W = img_feat.size()

        # compute attention weights
        qst_feat_tiled = qst_feat.view(B, self.qst_feat_dim, 1, 1).expand(B, self.qst_feat_dim, H, W) # [B,q,h,w]
        concat_feat = torch.cat((qst_feat_tiled, img_feat), 1)
        att_emb = self.att_encoder(concat_feat) # [B, num_stacks, h, w]
        self.att_weights = self.att_softmax( \
                att_emb.view(B, self.num_stacks, H*W)).view(B, self.num_stacks, H, W)

        replicated_img_feat = img_feat.view(B, 1, K, H, W).expand(B, self.num_stacks, K, H, W)
        replicated_att_weights = \
            self.att_weights.view(B, self.num_stacks, 1, H, W).expand(B, self.num_stacks, K, H, W)
        ctxs = (replicated_img_feat * replicated_att_weights).sum(3).sum(3) # [B, s, K]
        ctxs = ctxs.view(B, self.num_stacks*K)

        return ctxs, self.att_weights

    def print_status(self, logger, prefix=""):
        for ns in range(self.num_stacks):
            logger.info(
                "{}-ATT-{}-stack max={:.6f} | min={:.6f}".format(
                    prefix, (ns+1), self.att_weights[0, ns].max().data[0],
                    self.att_weights[0, ns].min().data[0]))


class StackedAttention(nn.Module):
    def __init__(self, config):
        super(StackedAttention, self).__init__() # Must call super __init__()

        # get SAN configurations
        self.num_stacks = utils.get_value_from_dict(config, "num_stacks", 2)
        qst_feat_dim = utils.get_value_from_dict(config, "qst_emb_dim", 256)
        img_feat_dim = utils.get_value_from_dict(config, "img_emb_dim", 256)
        self.att_emb_dim = utils.get_value_from_dict(config, "att_emb_dim", 512)
        att_nonlinear = utils.get_value_from_dict(config, "att_nonlinear_fn", "None")

        assert self.num_stacks > 0, "# of stacks {} < 1.".format(self.num_stacks)
        if self.num_stacks > 1:
            assert qst_feat_dim == img_feat_dim

        # layers for 1st attention
        self.query_encoder_1 = get_linear(qst_feat_dim, self.att_emb_dim,
            bias=False, dropout=0, nonlinear=att_nonlinear)
        self.img_encoder_1 = get_conv2d(
            img_feat_dim, self.att_emb_dim, 1, 1, dropout=0, nonlinear=att_nonlinear)
        self.att_encoder_1 = get_conv2d(
            self.att_emb_dim, 1, 1, 1, dropout=0, nonlinear="None")
        self.att_softmax_1 = nn.Softmax(dim=2)

        # layers for after 2nd attention
        if self.num_stacks > 1:
            self.query_encoder_stack = nn.ModuleList()
            self.query_encoder_stack = nn.ModuleList()
            self.img_encoder_stack = nn.ModuleList()
            self.att_encoder_stack = nn.ModuleList()
            self.att_softmax_stack = nn.ModuleList()
            #self.avg2d_stack = nn.ModuleList()
            for si in range(self.num_stacks-1):
                self.query_encoder_stack.append( get_linear(
                    qst_feat_dim, self.att_emb_dim,
                    dropout=0, nonlinear=att_nonlinear)
                )
                self.img_encoder_stack.append( get_conv2d(
                    img_feat_dim, self.att_emb_dim, 1, 1, bias=False,
                    dropout=0, nonlinear=att_nonlinear)
                )
                self.att_encoder_stack.append( get_conv2d(
                    self.att_emb_dim, 1, 1, 1, dropout=0, nonlinear="None")
                )
                self.att_softmax_stack.append(nn.Softmax(dim=2))

    def forward(self, qst_feat, img_feat):
        """ Compute context vector given qst feature and visual features
        Args:
            qst_feat: [B, qst_feat_dim]
            img_feat: [B, img_feat_dim, h, w]
        Returns:
            ctx_feat: [B, ctx_feat_dim]
            att_weights_list: list of attention weithgs [(B, h, w), ..., (B, h, w)]
        """

        self.att_weights_list = []
        B, K, H, W = img_feat.size()

        # compute 1st attention weights
        query_emb_1 = self.query_encoder_1(qst_feat) # [B, att_dim]
        img_emb_1 = self.img_encoder_1(img_feat) # [B, att_dim, h, w]
        att_emb_1 = self.att_encoder_1( \
                F.tanh(img_emb_1 + query_emb_1.view(
                    B, self.att_emb_dim, 1, 1).expand_as(img_emb_1))) # [B,1,h,w]
        # [B,1,h,w] -> [B,1,h*w] -> [B,1,h,w]
        att_weights_1 = self.att_softmax_1(att_emb_1.view(B, 1, H*W)).view(B, 1, H, W)

        # compute 1st context vectors (weighted sum)
        ctx_1 = img_feat * att_weights_1.expand_as(img_feat) # [B, img_dim, h, w]
        ctx_1 = ctx_1.sum(2).sum(2)
        #F.avg_pool2d(ctx_1, H) # [B, img_feat_dim, 1, 1]

        # query vector where initial query vector is a question feature
        query_feat = qst_feat + ctx_1.squeeze() # [B, img_feat_dim]
        # append attention weights
        self.att_weights_list.append(att_weights_1.squeeze().clone())

        # compute after 2nd attention weights (after stack 2)
        if self.num_stacks > 1:
            for si in range(self.num_stacks-1):
                # compute attention weights
                query_emb_stack = self.query_encoder_stack[si](query_feat) # [B, att_dim]
                img_emb_stack = self.img_encoder_stack[si](img_feat) # [B, att_dim, h, w]
                att_emb_stack = self.att_encoder_stack[si](F.tanh(img_emb_stack \
                        + query_emb_stack.view(B, self.att_emb_dim, 1, 1).expand_as(img_emb_stack)))
                att_weights_stack = self.att_softmax_stack[si](
                    att_emb_stack.view(B, 1, H*W)).view(B, 1, H, W)

                # compute context vectors (weighted sum)
                ctx_stack = img_feat * att_weights_stack.expand_as(img_feat) # [B, img_dim, h, w]
                ctx_stack = ctx_stack.sum(2).sum(2)
                #F.avg_pool2d(ctx_stack, H) # [B, img_dim, 1, 1]

                # compute query vector
                query_feat = query_feat + ctx_stack.squeeze()
                # append attention weights
                self.att_weights_list.append(att_weights_stack.squeeze().clone())

        return query_feat, self.att_weights_list

    def print_status(self, logger, prefix=""):
        for ns in range(self.num_stacks):
            logger.info(
                "{}-ATT-{}-stack max={:.6f} | min={:.6f}".format(
                    prefix, (ns+1), self.att_weights_list[ns][0].max().data[0],
                    self.att_weights_list[ns][0].min().data[0]))

"""
Layers for loss
"""
class EnsembleLoss(nn.Module):
    def __init__(self, config):
        super(EnsembleLoss, self).__init__() # Must call super __init__()

        self.logger = io_utils.get_logger("Train")

        self.use_gpu = utils.get_value_from_dict(config, "use_gpu", True)
        self.use_assignment_model = utils.get_value_from_dict(
            config, "use_assignment_model", False)
        self.beta = utils.get_value_from_dict(config, "beta", 0.75)
        self.k = utils.get_value_from_dict(config, "num_overlaps", 2)
        self.m = utils.get_value_from_dict(config, "num_models", 3)
        self.tau  = utils.get_value_from_dict(config, "tau", -1)
        self.num_labels = utils.get_value_from_dict(config, "num_labels", 28)
        self.use_initial_assignment = \
                utils.get_value_from_dict(config, "use_initial_assignment", False)
        self.use_KD_loss_with_ensemble = \
                utils.get_value_from_dict(config, "use_KD_loss_with_ensemble", False)
        self.assign_using_accuracy = \
                utils.get_value_from_dict(config, "assign_using_accuracy", False)
        self.version = utils.get_value_from_dict(config, "version", "KD-MCL")
        self.use_ensemble_loss = utils.get_value_from_dict(
            config, "use_ensemble_loss", False)

        if self.use_assignment_model:
            self.assignment_criterion = nn.CrossEntropyLoss()

        self.iteration = 0

    def forward(self, inp, labels):
        """
        Args:
            inp: list of two items: logit list (m * [B, C]) and CrossEntropyLoss list (m * [B,])
            labels: answer labels [B,]
        Returns:
            loss: scalar value
        """

        # for all answers, we just use most frequent answers as ground-truth
        if type(labels) == type(list()):
            labels = labels[0]
        B = labels.size(0)
        logit_list = inp[0]
        ce_loss_list = inp[1]
        ce_loss_tensor = torch.stack(ce_loss_list, dim=0) # [m,B]
        assert (ce_loss_tensor.dim() == 2) and (ce_loss_tensor.size(1) == B), \
                "Loss of base network should not be reduced (m, B)"

        log_txt = ""
        self.assignments = None
        if self.use_assignment_model and self.version != "IE":
            assignment_logits = inp[2]

        if self.version == "IE":
            # naive independent ensemble learning
            txt = "IE "
            print(txt, end="\r")
            if (self.iteration % (500)) == 0:
                self.logger.info(txt)

            total_loss = sum(ce_loss_list)
            total_loss = total_loss.sum() / self.m / B
            # End of IE

        elif self.version == "sMCL":
            # stochastic MCL (sMCL) [Stefan Lee et. al., 2016] (https://arxiv.org/abs/1606.07839)
            txt = "sMCL "
            print(txt, end="")
            if (self.iteration % (500)) == 0:
                log_txt += txt
            min_val, min_idx = ce_loss_tensor.t().topk(self.k, dim=1, largest=False)
            self.assignments = net_utils.get_data(min_idx) # [B,k]
            min_idx = min_idx.t()
            total_loss = min_val.sum() / B
            # End of sMCL

        elif self.version == "KD-MCL":
            """ obtain assignments (sorting oracle loss) """
            # compute KL divergence with teacher distribution for each model
            KLD_list = self.compute_kld(logit_list, inp[-1])

            # compute or get assignments
            if self.use_initial_assignment:
                txt = "KD-MCL using pre-computed assignments "
                print(txt, end="")
                if (self.iteration % (500)) == 0:
                    log_txt += txt
                assignments = inp[-2].clone()
                if assignments.dim() == 1:
                    min_idx = assignments.clone().view(1, assignments.size(0))
                elif assignments.dim() == 2:
                    min_idx = assignments.clone().t() # [k, B]
                else:
                    raise ValueError("Error: dim(assignments) > 2")
                self.assignments = assignments.cpu().data # [B, k]

                # re-assign based on accuracy
                # for assigned data, only when assigned model is failure and
                # other model predicts correctly, we replace the assignment
                if self.assign_using_accuracy:
                    prob_list = [F.softmax(logit, dim=1).clamp(1e-10, 1.0) \
                                 for logit in logit_list] # m*[B,C]
                    indices= []
                    corrects = []
                    for prob in prob_list:
                        val, idx = torch.max(prob, dim=1)
                        indices.append(idx)
                        corrects.append(torch.eq(idx, labels))
                    correct_tensors = torch.stack(corrects, dim=0).data.cpu() # [m,B]
                    selected_corrects = correct_tensors.gather(0, min_idx)
                    new_assignments = np.zeros((B, self.k))
                    origin_assignments = self.assignments.numpy() # [B,k]
                    num_changed = 0
                    for b in range(B):
                        if (selected_corrects[:,b].sum() == 0) \
                                and (correct_tensors[:,b].sum() > 0):
                            num_changed += 1
                            replace_cands = []
                            for s in range(self.m):
                                if correct_tensors[s,b] == 1:
                                    replace_cands.append(s)
                            num_cands = len(replace_cands)
                            if num_cands  >= self.k:
                                shuffle(replace_cands)
                                new_assignments[b] = np.asarray(replace_cands[:self.k])
                            else:
                                new_assignments[b,:num_cands] = \
                                    np.asarray(replace_cands)
                                new_assignments[b,num_cands:] = \
                                    origin_assignments[b,:self.k-num_cands]
                        else:
                            new_assignments[b] = origin_assignments[b]

                    # overwrite assignments
                    self.assignments = torch.from_numpy(new_assignments).long()
                    min_idx = Variable(self.assignments.clone().t())
                    txt = "{:03d} changed ".format(num_changed)
                    print(txt, end="")
                    if (self.iteration % (500)) == 0:
                        log_txt += txt

            else:
                # compute oracle loss
                oracle_loss_list = []
                for mi in range(self.m):
                    if (mi+1) != self.m:
                        oracle_loss_list.append( ce_loss_list[mi] \
                                + self.beta * (sum(KLD_list[:mi]) \
                                + sum(KLD_list[mi+1:]))) # m * [B,]
                    else:
                        oracle_loss_list.append(ce_loss_list[mi] \
                                + self.beta * sum(KLD_list[:mi])) # m*[B]

                # compute assignments
                oracle_loss_tensor = torch.stack(oracle_loss_list, dim=0) # [m,B]
                min_val, min_idx = oracle_loss_tensor.t().topk(
                        self.k, dim=1, largest=False) # [B,k]
                self.assignments = net_utils.get_data(min_idx) # [B,k]
                min_idx = min_idx.t() # [k,B]

                KLD_tensor = torch.stack(KLD_list, dim=0) # [m,B]
                txt = "KD-MCL CE {} | KLD {} | ORACLE {} ".format(
                    "/".join("{:.3f}".format(
                        ce_loss_tensor[i,0].data[0]) for i in range(self.m)),
                    "/".join("{:6.3f}".format(
                        KLD_tensor[i,0].data[0]) for i in range(self.m)),
                    "/".join("{:6.3f}".format(
                        oracle_loss_tensor[i,0].data[0]) for i in range(self.m)))
                print(txt, end="")
                if (self.iteration % (500)) == 0:
                    log_txt += txt

            # compute loss with assignments
            total_loss = 0
            for mi in range(self.m):
                for topk in range(self.k):
                    selected_mask = min_idx[topk].eq(mi).float() # [B]
                    if self.use_gpu and torch.cuda.is_available():
                        selected_mask = selected_mask.cuda()

                    cur_loss = net_utils.where(selected_mask,
                            ce_loss_list[mi], self.beta*KLD_list[mi])
                    total_loss += cur_loss.sum()
            #total_loss = total_loss / self.m / B / self.k
            total_loss = total_loss / B
            # End of KD-MCL

        elif self.version == "ALL":
            txt = "ALL loss "
            print(txt, end="")
            if (self.iteration % (500)) == 0:
                log_txt += txt
            # compute KL divergence with teacher distribution for each model
            KLD_list = self.compute_kld(logit_list, inp[-1])
            # compute KL divergence with Uniform distribution for each model
            prob_list = [F.softmax(logit, dim=1).clamp(1e-10, 1.0)
                         for logit in logit_list] # m*[B,C]
            ENT_list = [(-prob.log().mean(dim=1)).add(-np.log(self.num_labels))
                            for prob in prob_list] # m*[B]

            # compute oracle loss
            oracle_loss_list = []
            for mi in range(self.m):
                if (mi+1) != self.m:
                    oracle_loss_list.append( ce_loss_list[mi] \
                        + self.beta * (sum(KLD_list[:mi])+sum(KLD_list[mi+1:])) \
                        + 0.5 * (sum(ENT_list[:mi])+sum(ENT_list[mi+1:])) \
                    ) # m * [B,]
                else:
                    oracle_loss_list.append(ce_loss_list[mi] \
                        + self.beta * sum(KLD_list[:mi]) \
                        + 0.5 * sum(ENT_list[:mi])
                    ) # m*[B]

            # compute assignments
            oracle_loss_tensor = torch.stack(oracle_loss_list, dim=0) # [m,B]
            min_val, min_idx = oracle_loss_tensor.t().topk(
                    self.k, dim=1, largest=False) # [B,k]
            self.assignments = net_utils.get_data(min_idx) # [B,k]
            min_idx = min_idx.t() # [k,B]

            # compute loss with assignments
            total_loss = 0
            for mi in range(self.m):
                for topk in range(self.k):
                    selected_mask = min_idx[topk].eq(mi).float() # [B]
                    if self.use_gpu and torch.cuda.is_available():
                        selected_mask = selected_mask.cuda()

                    cur_loss = net_utils.where(selected_mask,
                            ce_loss_list[mi], self.beta*KLD_list[mi]+0.5*ENT_list[mi])
                    total_loss += cur_loss.sum()
            #total_loss = total_loss / self.m / B / self.k
            total_loss = total_loss / B
            # end of ALL

        else:
            """ compute assignments (sorting confident oracle loss) """
            # compute KL divergence with Uniform distribution for each model
            prob_list = [F.softmax(logit, dim=1).clamp(1e-10, 1.0)
                         for logit in logit_list] # m*[B,C]
            ENT_list = [(-prob.log().mean(dim=1)).add(-np.log(self.num_labels))
                            for prob in prob_list] # m*[B]
            oracle_loss_list = []
            for mi in range(self.m):
                if (mi+1) != self.m:
                    oracle_loss_list.append( ce_loss_list[mi] \
                        + self.beta*(sum(ENT_list[:mi])+sum(ENT_list[mi+1:]))) # m * [B,]
                else:
                    oracle_loss_list.append(ce_loss_list[mi] \
                            + self.beta*sum(ENT_list[:mi])) # m*[B]

            oracle_loss_tensor = torch.stack(oracle_loss_list, dim=0) # [m,B]
            entropy_tensor = torch.stack(ENT_list, dim=0) # [m,B]

            min_val, min_idx = oracle_loss_tensor.t().topk(self.k, dim=1, largest=False) # [B,k]
            self.assignments = net_utils.get_data(min_idx) # [B,k]
            min_idx = min_idx.t() # [k,B]

            if self.version == "CMCL_v0":
                txt = "CMCL v0 "
                print(txt, end="")
                if (self.iteration % (500)) == 0:
                    log_txt += txt
                total_loss = 0
                for mi in range(self.m):
                    for topk in range(self.k):
                        selected_mask = min_idx[topk].eq(mi).float() # [B]
                        if self.use_gpu and torch.cuda.is_available():
                            selected_mask = selected_mask.cuda()

                        total_loss += net_utils.where(selected_mask,
                                ce_loss_list[mi], self.beta*ENT_list[mi]).sum()
                total_loss += (self.beta*sum(ENT_list)).sum()
                total_loss = total_loss / self.m / B
                # end of CMCL v0

            elif self.version == "CMCL_v1":
                txt = "CMCL v1 "
                print( txt, end="")
                if (self.iteration % (500)) == 0:
                    log_txt += txt
                # sampling stochastically labels
                np_random_labels = np.random.randint(0, self.num_labels, size=(self.m, B))
                random_labels = Variable(
                    torch.from_numpy(np_random_labels), requires_grad=False).long() # [m, B]
                one_mask = Variable(
                    torch.from_numpy(np.ones(B)).float(), requires_grad=False) # [B]
                beta_mask = Variable(
                    torch.from_numpy(np.full((B), self.beta)).float(), requires_grad=False) # [B]

                if self.use_gpu and torch.cuda.is_available():
                    random_labels = random_labels.cuda()
                    one_mask = one_mask.cuda()
                    beta_mask = beta_mask.cuda()

                # compute loss
                for mi in range(self.m):
                    for topk in range(self.k):
                        selected_mask = min_idx[topk].eq(mi).long() # [B]
                        if self.use_gpu and torch.cuda.is_available():
                            selected_mask = selected_mask.cuda()

                        if topk == 0:
                            sampled_labels = net_utils.where(selected_mask, labels, random_labels[mi])
                            coeffi = selected_mask
                        else:
                            sampled_labels = net_utils.where(selected_mask, labels, sampled_labels)
                            coeffi += selected_mask

                    finally_selected = coeffi.ge(1.0).float()
                    coeff = net_utils.where(finally_selected, one_mask, beta_mask)
                    loss = coeff * F.cross_entropy(logit_list[mi], sampled_labels, reduce=False)
                    if mi == 0:
                        #total_loss = (loss.sum()) / self.m / B / self.k # 2017.3.20 11:13 pm
                        #total_loss = (loss.sum()) / B / self.k # 2017.3.22 08:16 pm
                        total_loss = loss.sum() / self.m / B
                    else:
                        #total_loss += (loss.sum()) / self.m / B / self.k # 2017.3.20 11:13 pm
                        #total_loss += (loss.sum()) / B / self.k # 2017.3.22 08:16 pm
                        total_loss += (loss.sum() / B / self.m)
                # end of CMCL v1

            else:
                raise NotImplementedError("Not supported version for CMCL Loss %s" % self.version)

            txt = "CE {} || ENT {} | ORACLE {} ".format(
                "/".join("{:.3f}".format(
                    ce_loss_tensor[i,0].data[0]) for i in range(self.m)),
                "/".join("{:6.2f}".format(
                    entropy_tensor[i,0].data[0]) for i in range(self.m)),
                "/".join("{:6.2f}".format(
                    oracle_loss_tensor[i,0].data[0]) for i in range(self.m)))
            print(txt, end="")
            if (self.iteration % (500)) == 0:
                log_txt += txt

        if self.use_ensemble_loss:
            logits = torch.stack(logit_list, 0).mean(dim=0)
            probs = F.softmax(logits, dim=1)
            ensemble_loss = F.cross_entropy(probs, labels)
            total_loss += ensemble_loss
            txt = "EL {:.3f} ".format(ensemble_loss.data[0])
            print(txt, end="")
            if (self.iteration % (500)) == 0:
                log_txt += txt

        if self.version != "IE":
            txt = "|".join("{:03d}".format(
                min_idx[0].eq(i).sum().data[0])
                for i in range(self.m))
            print(txt, end="\r")
            if (self.iteration % (500)) == 0:
                self.logger.info(log_txt + txt)

        # learn to predict assignment using question features
        if self.use_assignment_model and (self.version != "IE"):
            assignment_gt = Variable(
                self.assignments[:, 0].clone().long(),
                requires_grad=False)
            if self.use_gpu and torch.cuda.is_available():
                assignment_gt = assignment_gt.cuda()

            self.assignment_loss = self.assignment_criterion(
                    assignment_logits, assignment_gt)
            total_loss += self.assignment_loss
        else:
            self.assignment_loss = None

        # increment the iteration number
        self.iteration += 1

        return total_loss

    def compute_kld(self, logit_list, teacher_list):
        # compute KL divergence with teacher distribution for each model
        if self.use_KD_loss_with_ensemble:
            print("Ensembled teacher logit ", end="")
            # logit version
            teacher_logit = sum(teacher_list) / self.m
            KLD_list = [net_utils.compute_kl_div( \
                student_logit, teacher_logit, self.tau) \
                for student_logit in logit_list] # m*[B]
            # prob version
            teacher_prob = sum([F.softmax(tl, dim=1) for tl in teacher_list]) / self.m
            KLD_list = [net_utils.compute_kl_div( \
                student_logit, teacher_prob, self.tau, False) \
                for student_logit in logit_list] # m*[B]
        else:
            KLD_list = [net_utils.compute_kl_div( \
                student_logit, teacher_logit, self.tau) \
                for student_logit, teacher_logit \
                in zip(logit_list, teacher_list)] # m*[B]
        return KLD_list


class KLDLoss(nn.Module):
    def __init__(self, config):
        super(KLDLoss, self).__init__() # Must call super __init__()

        self.tau = utils.get_value_from_dict(config["model"], "tau", 1)
        self.reduce = utils.get_value_from_dict(config["model"], "loss_reduce", True)
        self.apply_softmax_on_teacher = True

    def forward(self, logits, gts):
        # This method accepts ground-truth as second argument
        # to be consistent with VirtualNetwork Class.
        # Thus, we do not use gts to compute KL divergence loss.
        student = logits[0]
        teacher = logits[1]
        return net_utils.compute_kl_div(student, teacher,
                    self.tau, self.apply_softmax_on_teacher, self.reduce)



class MultipleCriterion(nn.Module):
    def __init__(self, criterion=nn.CrossEntropyLoss(), loss_reduce=True):
        super(MultipleCriterion, self).__init__() # Must call super __init__()

        self.criterion = criterion
        self.criterion.reduce = False
        self.loss_reduce = loss_reduce

    def forward(self, inp, labels):
        """
        Args:
            inp: logits or probabilities [B, C]
            labels: list of [answers, all_answers, mask]
                - answers: do not use this
                - all_answers: all answers [B, A]
                - mask: mask of answers [B, A]
        """
        all_labels = labels[1]
        mask = labels[2]
        _, C = inp.size()
        B, A = all_labels.size()

        flat_inp = inp.view(B,1,C).expand(B,A,C).contiguous().view(B*A,C)
        flat_all_labels = all_labels.view(B*A)

        loss = self.criterion(flat_inp, flat_all_labels)
        loss = loss * mask.view(B*A)
        loss = loss.view(B,A).mean(dim=1)

        if self.loss_reduce:
            loss = loss.mean()

        return loss
