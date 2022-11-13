"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, label_similarity, support_features, support_labels=None, mask=None, contrastive="Normal"):
        """Compute loss for model. If both `support_labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            support_features: hidden vector of shape [bsz, n_views, ...].
            query_features: [bsz, n_views, d]
            query_labels: [bsz]
            support_labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if support_features.is_cuda
                  else torch.device('cpu'))

        if len(support_features.shape) < 3:
            raise ValueError('`support_features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(support_features.shape) > 3:
            support_features = support_features.view(support_features.shape[0], support_features.shape[1], -1)

        batch_size = support_features.shape[0]
        if support_labels is not None and mask is not None:
            raise ValueError('Cannot define both `support_labels` and `mask`')
        elif support_labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif support_labels is not None:
            support_labels = support_labels.contiguous().view(-1, 1)
            if support_labels.shape[0] != batch_size:
                raise ValueError('Num of support_labels does not match num of support_features')
            mask = torch.eq(support_labels, support_labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = support_features.shape[1]
        contrast_feature = torch.cat(torch.unbind(support_features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = support_features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 分配各负类权重
        if contrastive != "Normal":
            flat_similarity = label_similarity.view(-1).to(device)
            index = (support_labels * support_labels.T).view(-1).to(device)
            weight_matrix = torch.index_select(flat_similarity, 0, index).view(logits_mask.shape).to(device)
            logits_mask = weight_matrix * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # avoid nan loss when there's one sample for a certain class, e.g., 0,1,...1 for bin-cls , this produce nan for 1st in Batch
        # which also results in batch total loss as nan. such row should be dropped
        pos_per_sample = mask.sum(1)    # B
        pos_per_sample[pos_per_sample < 1e-6] = 1.0
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample   # mask.sum(1)

        #mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        """
        # 从这里开始，计算query和support之间的对比损失
        support_labels = support_labels.view(-1, 1)
        query_labels = query_labels.view(-1, 1)
        cross_mask = torch.eq(query_labels, support_labels.T).float().to(device)
        # compute logits
        query_features = torch.cat(torch.unbind(query_features, dim=1), dim=0)
        support_features = torch.cat(torch.unbind(support_features, dim=1), dim=0)
        query_dot_contrast = torch.div(
            torch.matmul(query_features, support_features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(query_dot_contrast, dim=1, keepdim=True)
        logits = query_dot_contrast - logits_max.detach()
        # 计算log
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        pos_per_sample = cross_mask.sum(1)    # B
        pos_per_sample[pos_per_sample < 1e-6] = 1.0
        mean_log_prob_pos = (cross_mask * log_prob).sum(1) / pos_per_sample   # mask.sum(1)
        cross_loss = -mean_log_prob_pos
        cross_loss = cross_loss.view(anchor_count, batch_size).mean()

        # print("loss: ", loss.item())
        # print("cross_loss: ", cross_loss.item())
        """

        return loss


class ProtoConLoss(nn.Module):
    """Prototypical Contrastive Learning
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ProtoConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, query_features, query_labels, prototypes, n_other, contrastive="Normal"):
        """Compute loss for model. If both `support_labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        Args:
            support_features: hidden vector of shape [bsz, feature_size].
            query_features: [bsz, feature_size]
            query_labels: [bsz]
            prototypes: [N, feature_size].
            n_other: other类对应几个向量
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if query_features.is_cuda
                  else torch.device('cpu'))

        batch_size = query_features.shape[0]
        prototype_labels = torch.cat([torch.zeros(n_other), torch.arange(1, prototypes.shape[0] - n_other + 1)]).view(-1, 1).to(device)    # [N + O + 1, 1]
        query_labels = query_labels.contiguous().view(-1, 1)
        if query_labels.shape[0] != batch_size:
            raise ValueError('Num of support_labels does not match num of support_features')
        mask = torch.eq(query_labels, prototype_labels.T).float().to(device)

        """
        for i in range(batch_size):
            for j in range(prototype_labels.shape[0]):
                mask[i][j] = label_similarity[query_labels[i][0]][query_labels[j][0]]
        """

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(query_features, prototypes.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        if contrastive != "Normal":
            # 分配各负类权重
            """
            flat_similarity = label_similarity.view(-1).to(device)
            index = (query_labels * prototype_labels.T).view(-1).to(device)
            weight_matrix = torch.index_select(flat_similarity, 0, index).view(mask.shape).to(device)
            exp_logits = torch.exp(logits) * weight_matrix
            """
            exp_logits = torch.exp(logits)
        else:
            exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # avoid nan loss when there's one sample for a certain class, e.g., 0,1,...1 for bin-cls , this produce nan for 1st in Batch
        # which also results in batch total loss as nan. such row should be dropped
        # print("mask: ", mask.shape)
        # print("log_prob: ", log_prob.shape)
        pos_per_sample = mask.sum(1)    # B
        pos_per_sample[pos_per_sample < 1e-6] = 1.0
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample   # mask.sum(1)

        #mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        return loss


def distance(x_1, x_2):
    return 1 - F.cosine_similarity(x_1, x_2, dim=0)


def gaussian(dist, sigma=1.0):
    return math.e ** (-dist ** 2 / 2 / sigma ** 2)
