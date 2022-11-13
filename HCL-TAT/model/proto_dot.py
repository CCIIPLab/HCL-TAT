# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from contrastive_loss import SupConLoss, ProtoConLoss, distance, gaussian


def dropout_augmentation(token_ids):
    return NotImplementedError


class ProtoDot(nn.Module):

    def __init__(self, encoder, opt):
        super(ProtoDot, self).__init__()

        self.feature_size = opt.feature_size
        self.max_len = opt.max_length
        self.distance_metric = opt.distance_metric

        self.encoder = encoder
        # self.encoder = nn.DataParallel(self.encoder)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.feature_size, self.feature_size, bias=False)
        )
        self.similarity_mlp = nn.Sequential(
            nn.Linear(opt.trainN + 1, opt.trainN + 1),
            nn.ReLU(),
            nn.Linear(opt.trainN + 1, opt.trainN + 1),
            nn.Sigmoid()
        )

        # self attention
        self.Wk = nn.Linear(self.feature_size, self.feature_size)
        self.Wq = nn.Linear(self.feature_size, self.feature_size)
        self.Wv = nn.Linear(self.feature_size, self.feature_size)

        with torch.no_grad():
            pad_embedding = self.encoder(torch.LongTensor([[0]]))[0].view(self.feature_size)
            pad_embedding = pad_embedding.repeat(opt.O, 1)
        # self.other_prototype = nn.Parameter(torch.randn(opt.O, self.feature_size))

        self.cost = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        self.contrastive_cost = SupConLoss(temperature=opt.temperature_alpha)
        self.proto_contrastive_cost = ProtoConLoss(temperature=opt.temperature_beta)

        self.dropout_rate = opt.dropout
        self.drop = nn.Dropout(self.dropout_rate)
        self.use_BIO = opt.use_BIO
        self.O = opt.O
        self.contrastive = opt.contrastive
        self.alpha = opt.alpha
        self.beta = opt.beta
        self.threshold = opt.threshold

    def get_embedding(self, support_set, query_set):
        # encode
        support_emb = self.encoder(support_set['tokens'], attention_mask=support_set["att-mask"])[
            0]  # B*N*K, max_len, feature_size
        query_emb = self.encoder(query_set['tokens'], attention_mask=query_set['att-mask'])[
            0]  # B*N*K, max_len, feature_size

        # dropout
        support_emb = self.drop(support_emb)  # B*N*K, max_len, feature_size
        query_emb = self.drop(query_emb)  # B*N*K, max_len, feature_size
        return support_emb, query_emb

    def forward(self, support_set, query_set, N, K, Q, O):
        support_emb, query_emb = self.get_embedding(support_set, query_set)

        support_emb = support_emb.view(-1, N, K, self.max_len, self.feature_size)  # B, N, K, max_len, feature_size
        query_emb = query_emb.view(-1, N * Q, self.max_len, self.feature_size)  # B, N*Q, max_len, feature_size

        B_mask = support_set['B-mask'].view(-1, N, K, self.max_len)  # B, N, K, max_len
        I_mask = support_set['I-mask'].view(-1, N, K, self.max_len)  # B, N, K, max_len
        text_mask = support_set['text-mask'].view(-1, N, K, self.max_len)  # B, N, K, max_len

        # prototype 
        prototype = self.proto(support_emb, B_mask, I_mask, text_mask)  # B, 2*N+1, feature_size

        # classification
        logits, other_max_sim_index = self.similarity(prototype, query_emb, N, K, Q, O)  # B, N*Q, max_len, 2*N+1
        _, pred = torch.max(logits.view(-1, logits.shape[-1]), 1)  # B*N*Q*max_len

        outputs = (logits, pred, support_emb, query_emb, prototype)

        # loss
        if query_set['trigger_label'] is not None:
            loss = self.loss(logits, query_set['trigger_label'])
            if self.contrastive != "None":
                support_emb_for_contrast = self.projection_head(support_emb).view(-1, self.feature_size)
                query_emb_for_contrast = self.projection_head(query_emb).view(-1, self.feature_size)
                prototype_for_contrast = self.projection_head(prototype)
                # support_emb_aug, query_emb_aug = self.get_embedding(support_set, query_set)
                # support_emb_for_contrast = self.projection_head(torch.cat([support_emb.view(support_emb_aug.shape), support_emb_aug]))
                # query_emb_for_contrast = self.projection_head(torch.cat([query_emb, query_emb_aug]))
                prototype = prototype.view(-1, self.feature_size)
                l2_prototype = F.normalize(prototype, p=2, dim=1)
                # label_similarity = 1 - torch.matmul(l2_prototype, l2_prototype.T)
                # label_similarity = torch.exp(label_similarity * label_similarity / -2)
                # label_similarity = torch.matmul(l2_prototype, l2_prototype.T)
                # label_similarity = self.similarity_mlp(label_similarity)
                label_similarity = torch.ones(N + 1, N + 1).to(prototype)
                label_similarity[0, :] = 2.0
                label_similarity[:, 0] = 2.0
                label_similarity[0][0] = 1.0
                # print("label_similarity: ", label_similarity)
                contrastive_loss = self.contrastive_loss(torch.cat([support_emb_for_contrast]),
                                                         torch.cat([support_set["trigger_label"]]),
                                                         label_similarity, self.contrastive)
                proto_contrastive_loss = self.prototypical_contrastive_loss(query_emb_for_contrast,
                                                                            query_set["trigger_label"],
                                                                            prototype_for_contrast, label_similarity,
                                                                            other_max_sim_index,
                                                                            self.contrastive)
                outputs = (loss + self.alpha * contrastive_loss + self.beta * proto_contrastive_loss,) + outputs
            else:
                outputs = (loss,) + outputs

        return outputs

    def similarity(self, prototype, query, N, K, Q, O):
        '''
        inputs:
            prototype: B, 2*N+1, feature_size或者B, N+1, feature_size
            query: B, N*Q, max_len, feature_size
        outputs:
            sim: B, N*Q, max_len, 2*N+1或者B, N*Q, max_len, N+1
        '''
        tag_num = prototype.shape[1]
        query_num = query.shape[1]

        query = query.unsqueeze(-2)  # B, N*Q, max_len, 1, feature_size
        query = query.expand(-1, -1, -1, tag_num, -1)  # B, N*Q, max_len, 2*N+1/N+1, feature_size

        prototype = prototype.unsqueeze(1)  # B, 1, 2*N+1, feature_size
        prototype = prototype.unsqueeze(2)  # B, 1, 1, 2*N+1, feature_size
        prototype = prototype.expand(-1, query_num, self.max_len, -1, -1)  # B, N*Q, max_len, 2*N+1/N+1, feature_size

        if self.distance_metric == "dot":
            sim = (prototype * query).sum(-1)  # B, N*Q, max_len, 2*N+O/N+O
        elif self.distance_metric == "match":
            sim = 1e3 * F.cosine_similarity(prototype, query, dim=-1)  # B, N*Q, max_len, 2*N+O/N+O
        else:
            sim = -(torch.pow(query - prototype, 2)).sum(-1)  # B, N*Q, max_len, 2*N+O/N+O

        # 对于前O个Other类的prototype，选择最高的sim作为logit
        B, NQ, max_len, _ = sim.shape
        if self.use_BIO:
            new_sim = torch.zeros(B, NQ, max_len, 2 * N + 1)
        else:
            new_sim = torch.zeros(B, NQ, max_len, N + 1)
        new_sim = new_sim.to(sim)
        if self.O > 0:
            new_sim[:, :, :, 0] = torch.max(torch.mean(sim[:, :, :, :O + 1].view(-1, O + 1), dim=0))
            other_max_sim_index = torch.argmax(sim[:, :, :, :O + 1])
        else:
            if self.threshold == "mean":
                new_sim[:, :, :, 0] = torch.mean(sim[:, :, :, 0])
            elif self.threshold == "max":
                new_sim[:, :, :, 0] = torch.max(sim[:, :, :, 0])
            else:
                new_sim[:, :, :, 0] = sim[:, :, :, 0]
            # new_sim[:, :, :, 0] = sim[:, :, :, 0]
            other_max_sim_index = 0
        new_sim[:, :, :, 1:] = sim[:, :, :, O + 1:]

        return new_sim, other_max_sim_index

    def proto(self, support_emb, B_mask, I_mask, text_mask):
        '''
        input:
            support_emb : B, N, K, max_len, feature_size
            B_mask : B, N, K, max_len
            I_mask : B, N, K, max_len
            att_mask: B, N, K, max_len
        output:
            prototype : B, 2*N+1, feature_size # (class_num -> 2N + 1)
        '''
        B, N, K, _, _ = support_emb.shape
        if self.use_BIO:
            prototype = torch.empty(B, 2 * N + self.O + 1, self.feature_size).to(support_emb)
        else:
            prototype = torch.empty(B, N + self.O + 1, self.feature_size).to(support_emb)

        B_mask = B_mask.unsqueeze(-1)
        B_mask = B_mask.expand(-1, -1, -1, -1, self.feature_size)
        B_mask = B_mask.to(support_emb)  # B, N, K, max_len, feature_size
        I_mask = I_mask.unsqueeze(-1)
        I_mask = I_mask.expand(-1, -1, -1, -1, self.feature_size)
        I_mask = I_mask.to(support_emb)  # B, N, K, max_len, feature_size
        text_mask = text_mask.unsqueeze(-1)
        text_mask = text_mask.expand(-1, -1, -1, -1, self.feature_size)
        text_mask = text_mask.to(support_emb)

        for i in range(B):
            O_mask = torch.ones_like(B_mask[i]).to(B_mask)  # N, K, max_len, feature_size
            O_mask -= B_mask[i] + I_mask[i]
            O_mask = O_mask * text_mask[i]
            for j in range(N):
                if self.use_BIO:
                    sum_B_fea = (support_emb[i, j] * B_mask[i, j]).view(-1, self.feature_size).sum(0)
                    num_B_fea = B_mask[i, j].sum() / self.feature_size + 1e-8
                    prototype[i, 2 * j + self.O + 1] = sum_B_fea / num_B_fea
                    sum_I_fea = (support_emb[i, j] * I_mask[i, j]).view(-1, self.feature_size).sum(0)
                    num_I_fea = I_mask[i, j].sum() / self.feature_size + 1e-8
                    prototype[i, 2 * j + self.O + 1 + 1] = sum_I_fea / num_I_fea
                else:
                    sum_I_fea = (support_emb[i, j] * I_mask[i, j]).view(-1, self.feature_size).sum(0)
                    num_I_fea = I_mask[i, j].sum() / self.feature_size + 1e-8
                    # print("num_I_fea: ", num_I_fea)
                    prototype[i, j + self.O + 1] = sum_I_fea / num_I_fea
            # 先用Other类的embedding计算一个prototype
            sum_O_fea = (support_emb[i] * O_mask).reshape(-1, self.feature_size).sum(0)
            num_O_fea = O_mask.sum() / self.feature_size + 1e-8
            prototype[i, 0] = sum_O_fea / num_O_fea
            # 对Other类做聚类
            # 如果存在辅助向量，则加到prototype后面
            if self.O >= 1:
                raise NotImplementedError("O must be set to 0")

        return prototype

    def proto_interaction(self, prototype):
        # self attention
        K = self.Wk(prototype)  # B, 2*N+1, feature_size
        Q = self.Wq(prototype)  # B, 2*N+1, feature_size
        V = self.Wv(prototype)  # B, 2*N+1, feature_size

        att_score = torch.matmul(K, Q.transpose(-1, -2))  # B, 2*N+1, 2*N+1
        att_score /= torch.sqrt(torch.tensor(self.feature_size).to(K))  # B, 2*N+1, 2*N+1
        att_score = att_score.softmax(-1)  # B, 2*N+1, 2*N+1

        prototype = torch.matmul(att_score, V)  # B, 2*N+1, feature_size
        return prototype

    def loss(self, logits, label):
        logits = logits.view(-1, logits.shape[-1])
        label = label.view(-1)

        loss_weight = torch.ones_like(label).float()
        loss = self.cost(logits, label)
        loss = (loss_weight * loss).mean()
        return loss

    def contrastive_loss(self, support_features, support_labels, label_similarity, contrastive="Normal"):
        """
        compute contrastive loss
        :param ignore_padding: 无视padding
        :param features: B, N, K, max_len, feature_size
        :param labels: B*N*K*max_len
        :return: loss
        """
        support_features = support_features.view(-1, 1, self.feature_size)
        support_labels = support_labels.view(-1)
        # query_features = support_features.view(-1, 1, self.feature_size)
        # query_labels = support_labels.view(-1)
        # delete Other and PAD label
        support_features = support_features[support_labels != -100].view(-1, 1, self.feature_size)
        support_labels = support_labels[support_labels != -100].view(-1)
        # print("support_features: ", support_features.shape)
        # print("support_labels: ", support_labels.shape)
        # support_features = support_features[support_labels != 0].view(-1, 1, self.feature_size)
        # support_labels = support_labels[support_labels != 0].view(-1)
        # query_features = query_features[query_labels != -100].view(-1, 1, self.feature_size)
        # query_labels = query_labels[query_labels != -100].view(-1)
        # query_features = query_features[query_labels != 0].view(-1, 1, self.feature_size)
        # query_labels = query_labels[query_labels != 0].view(-1)
        # L2 Normalize
        support_features = F.normalize(support_features, p=2, dim=2)
        # query_features = F.normalize(query_features, p= 2, dim=2)
        contrastive_loss = self.contrastive_cost(label_similarity, support_features, support_labels,
                                                 contrastive=contrastive)
        # print("contrastive_loss: ", contrastive_loss.item())
        return contrastive_loss

    def prototypical_contrastive_loss(self, query_features, query_labels, prototypes, label_similarity,
                                      other_max_sim_index,
                                      contrastive="Normal"):
        # Other类取相似度最高的那个prototype参与计算
        # print("other_max_sim_index: ", other_max_sim_index)
        prototypes = torch.cat([prototypes[:, 0: self.O + 1, :], prototypes[:, self.O + 1:, :]], dim=1)
        query_features = query_features.view(-1, self.feature_size)
        prototypes = prototypes.view(-1, self.feature_size)
        query_labels = query_labels.view(-1)
        query_features = query_features[query_labels != -100].view(-1, self.feature_size)
        query_labels = query_labels[query_labels != -100].view(-1)
        # L2 norm
        query_features = F.normalize(query_features, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        proto_contrastive_loss = self.proto_contrastive_cost(query_features, query_labels, prototypes, self.O + 1,
                                                             contrastive=contrastive)
        return proto_contrastive_loss
