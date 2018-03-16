


class DualAttention(nn.Module):
    def __init__(self, config):
        super(DualAttention, self).__init__() # Must call super __init__()

        # get SAN configurations
        self.num_stacks = utils.get_value_from_dict(config, "num_stacks", 2)
        qst_feat_dim = utils.get_value_from_dict(config, "qst_feat_dim", 512)
        img_feat_dim = utils.get_value_from_dict(config, "img_feat_dim", 256)
        self.att_dim = utils.get_value_from_dict(config, "att_emb_dim", 512)
        att_nonlinear = utils.get_value_from_dict(config, "att_nonlinear", "None")

        assert self.num_stacks > 0, \
            "The number of stacks in attention should be at least 1 (>{}).".format(self.num_stacks)
        if self.num_stacks > 1:
            assert qst_feat_dim == img_feat_dim

        self.query_encoder_1 = get_linear(
                qst_feat_dim, self.att_emb_dim, bias=False, dropout=0, nonlinear=att_nonlinear)
        self.img_encoder_1 = get_conv2d(
                img_feat_dim, self.att_emb_dim, 1, 1, dropout=0, nonlinear=att_nonlinear)
        self.att_encoder_1 = get_conv2d(
                self.att_emb_dim, 1, 1, 1, dropout=0, nonlinear="None")

        # define layers for visual attention
        self.Wv = get_conv2d(img_feat_dim, self.att_dim, 1, 1, nonlinear="Tanh")
        self.Wvm = get_linear(qst_feat_dim, self.att_dim, nonlinear="Tanh")
        self.Whv = get_conv2d(img_feat_dim, 1, 1, 1, nonlinear="None")
        self.vis_att_softmax = nn.Softmax(dim=2)
        self.P = get_linear(img_feat_dim, qst_feat_dim, nonlinear="Tanh")

        # define layers for visual attention
        self.Wu = get_conv1d(qst_feat_dim, self.att_dim, 1, 1, nonlinear="Tanh")
        self.Wum = get_linear(qst_feat_dim, self.att_dim, nonlinear="Tanh")
        self.Whu = get_conv1d(qst_feat_dim, 1, 1, 1)
        self.txt_att_softmax = nn.Softmax(dim=2)

    def forward(self, u, v, m):
        """ Compute context vector given qst feature and visual features
        Args:
            u: question feature [B, max_len, qst_feat_dim]
            v: image feature [B, img_feat_dim, h, w]
            m: query vector [B, qst_feat_dim]
        Returns:
            next_m: [B, ctx_feat_dim]
        """

        # get dimensions of each feature
        B, K, H, W = v.size()
        _, L, Q = u.size()
        A = self.att_dim

        # compute visual attention
        hv = self.Wv(v) * self.Wvm(m).view(B, A, 1, 1).expand(B, A, H, W) # [B, A, H, W]
        vis_att_weights = self.vis_att_softmax(self.Whv(hv).view(B, 1, H*W)).view(B, 1, H, W)
        v1 = self.P((v * vis_att_weights.expand_as(v)).sum(2).sum(2))

        # compute text attention
        u = u.transpose(1, 2) # [B, Q, L]
        hu = self.Wu(u) * self.Wum(m).view(B, A, 1).expand(B, A, L) # [B, A, L]
        txt_att_weights = self.txt_att_softmax(self.Whu(hu)) # [B, 1, L]
        u1 = (u * txt_att_weights.expand(B, Q, L)).sum(2).sum(2)

        next_m = m + (v1 * u1)

        self.vis_att_weights = vis_att_weights.clone().data.cpu()
        self.txt_att_weights = txt_att_weights.clone().data.cpu()

        return next_m

    def print_status(self, logger, prefix=""):
        logger.info(
            "{}-VIS-ATT max={:.6f} | min={:.6f} // {}-TXT-ATT max={:.6f} | min={:.6f}".format(
                prefix, self.vis_att_weights[0].max(), self.vis_att_weights[0].min(),
                prefix, self.txt_att_weights[0].max(), self.txt_att_weights[0].min(),
            ))


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, config):
        super(KnowledgeDistillationLoss, self).__init__() # Must call super __init__()

        self.logger = io_utils.get_logger("Train")

        self.use_gpu = utils.get_value_from_dict(config, "use_gpu", True)
        config = config["kd_loss"]
        self.model_number = utils.get_value_from_dict(config, "model_number", 0)
        self.num_labels = utils.get_value_from_dict(config, "num_labels", 28)
        self.use_precomputed_selection = \
                utils.get_value_from_dict(config, "use_precomputed_selection", False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inp, labels):
        """
        Args:
            inp: list; [student_logit, teacher_logit, selections]
            labels: answer labels [B,]
        Returns:
            loss: scalar value
        """

        B = labels.size(0)

        # compute student loss
        student_logit = inp[0]
        student_loss = self.criterion(student_logit, labels)

        # get teacher's distribution
        teacher_logit = Variable(inp[1].clone().data, requires_grad=False)
        teacher_prob = F.softmax(teacher_logit, dim=1)

        # compute KL divergence with teacher's distribution
        student_prob = F.log_softmax(student_logit, dim=1) # [B,C]
        KLD_loss = F.kl_div(student_prob, teacher_prob, reduce=False).sum(dim=1) # [B]

        # get selection
        self.selections = inp[2]
        min_idx = self.selections.t()

        # compute loss with selections
        total_loss = 0
        zero_loss = Variable(torch.zeros(student_loss.size()), requires_grad=False)
        if self.use_gpu and torch.cuda.is_available():
            zero_loss = zero_loss.cuda()

        for topk in range(self.selections.size(1)):
            selected_mask = min_idx[topk].eq(self.model_number).float() # [B]
            if self.use_gpu and torch.cuda.is_available():
                selected_mask = selected_mask.cuda()

            # for selected networks, we subtract KLD_loss to
            # make overall computation easier
            total_loss += self._where(selected_mask,
                    student_loss - self.beta*KLD_loss, zero_loss).sum()
        total_loss += (self.beta*sum(KLD_loss)).sum()
        total_loss = total_loss / B

        return total_loss

    def _where(self, cond, x1, x2):
        """ Differentiable equivalent of np.where (or tf.where)
            Note that type of three variables should be same.
        Args:
            cond: condition
            x1: selected value if condition is 1 (True)
            x2: selected value if condition is 0 (False)
        """
        return (cond * x1) + ((1-cond) * x2)
