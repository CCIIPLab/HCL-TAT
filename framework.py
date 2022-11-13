# -*- coding: utf-8 -*-

import time
import os
import pdb
import matplotlib.pyplot as plt
import nni
from tqdm import tqdm

import torch
from torch.nn import DataParallel
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from sklearn.manifold import TSNE
from transformers import BertTokenizer


class Framework:
    def __init__(self,
                 train_dataset=None,
                 dev_dataset=None,
                 test_dataset=None,
                 metric=None,
                 device=None,
                 opt=None):
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.metric = metric
        self.device = device
        self.opt = opt
        self.tokenizer = BertTokenizer.from_pretrained("/home/xxx/data/huggingface/bert-base-uncased")

    def to_device(self, inputs):
        for k in inputs.keys():
            inputs[k] = inputs[k].to(self.device)
        return inputs

    def save_model(self, model, save_ckpt):
        checkpoint = {'opt': self.opt}
        if isinstance(model, DataParallel):
            checkpoint['state_dict'] = model.module.state_dict()
        else:
            checkpoint['state_dict'] = model.state_dict()
        torch.save(checkpoint, save_ckpt)

    def load_model(self, model, load_ckpt):
        if os.path.isfile(load_ckpt):
            checkpoint = torch.load(load_ckpt)
            print(f"Successfully loaded checkpoint : {load_ckpt}")
        else:
            raise Exception(f"No checkpoint found at {load_ckpt}")

        load_state = checkpoint['state_dict']
        model_state = model.state_dict()
        for name, param in load_state.items():
            if name not in model_state:
                continue
            if param.shape != model_state[name].shape:
                print(f"In load model : {name} param shape not match")
                continue
            model_state[name].copy_(param)

    def evaluate(self,
                 model,
                 eval_epoch,
                 evalN, K, Q, O, train_step,
                 mode="dev",
                 load_ckpt=None):

        if load_ckpt is not None:
            print(f"loading checkpint {load_ckpt}")
            self.load_model(model, load_ckpt)
            print(f"checkpoint {load_ckpt} loaded")

        model.to(self.device)
        self.metric.reset()

        if mode == "dev":
            eval_dataset = self.dev_dataset
        elif mode == "test":
            eval_dataset = self.test_dataset
        elif mode == "train":
            eval_dataset = self.train_dataset

        model.eval()
        for i in tqdm(range(eval_epoch)):
            support_set, query_set, id2label = next(eval_dataset)
            with torch.no_grad():
                support_set, query_set = self.to_device(support_set), self.to_device(query_set)
                loss, logits, pred, support_emb, query_emb, prototype = model(support_set, query_set, evalN, K, Q, O)
            self.metric.update_state(pred, query_set['trigger_label'], id2label)
            # 打印结果
            """
            if i == 6:
                raw_text_id = query_set["tokens"][0]
                ground_truth_id = query_set["trigger_label"][0]
                raw_text = self.tokenizer.convert_ids_to_tokens(raw_text_id)
                raw_text = [c for c in raw_text if c != "[PAD]" and c != "[CLS]" and c != "[SEP]"]
                ground_truth_type, pred_type = [], []
                print("input: ", " ".join(raw_text))
                for gid, pid in zip(ground_truth_id.cpu().numpy().tolist(), pred.cpu().numpy().tolist()):
                    if gid != -100:
                        ground_truth_type.append(id2label[0][gid])
                        pred_type.append(id2label[0][pid])
                print("ground_truth: ", ground_truth_type)
                print("pred: ", pred_type)
                with open("case_study/hcl_tat.txt", "w", encoding="utf-8") as f:
                    f.write("input: " + " ".join(raw_text) + "\n")
                    f.write("ground_truth: " + ",".join(ground_truth_type) + "\n")
                    f.write("pred: " + ",".join(pred_type) + "\n")
            """
            # 画embedding图
            if i == 2 and self.opt.embedding_visualization_path is not None:
                if not os.path.exists(self.opt.embedding_visualization_path):
                    os.mkdir(self.opt.embedding_visualization_path)
                support_attention_mask = support_set["text-mask"].view(-1).cpu().detach().numpy()
                support_labels = support_set["trigger_label"].view(-1).cpu().detach().numpy()
                query_attention_mask = query_set["text-mask"].view(-1).cpu().detach().numpy()
                query_labels = query_set["trigger_label"].view(-1).cpu().detach().numpy()
                query_preds = pred.view(-1).cpu().detach().numpy()
                support_features = support_emb.view(-1, self.opt.feature_size).cpu().detach().numpy()
                query_features = query_emb.view(-1, self.opt.feature_size).cpu().detach().numpy()
                prototype_features = prototype.view(-1, self.opt.feature_size).cpu().detach().numpy()
                self.draw_embeddings(support_attention_mask, support_labels, query_attention_mask,
                                     query_labels, query_preds, support_features, query_features, prototype_features,
                                     os.path.join(self.opt.embedding_visualization_path,
                                                  self.opt.model + "_" + mode + "_" + str(train_step) + ".svg"), id2label[0], evalN)

        return self.metric.result(self.opt.use_BIO)

    def train(self,
              model,
              trainN, evalN, K, Q, O,
              optimizer,
              scheduler,
              train_epoch,
              eval_epoch,
              eval_step,
              load_ckpt=None,
              save_ckpt=None):

        if load_ckpt is not None:
            print(f"loading checkpint {load_ckpt}")
            self.load_model(model, load_ckpt)
            print(f"checkpoint {load_ckpt} loaded")

        scaler = GradScaler()
        model.to(self.device)

        best_p = 0
        best_r = 0
        best_f1 = 0
        best_epoch = 0
        no_increase_interval = 0
        for epoch in range(train_epoch):
            epoch_begin = time.time()

            # train
            model.train()
            support_set, query_set, id2label = next(self.train_dataset)
            support_set, query_set = self.to_device(support_set), self.to_device(query_set)

            optimizer.zero_grad()

            with autocast():
                loss, logits, pred, _, _, _ = model(support_set, query_set, trainN, K, Q, O)
                loss = loss.mean()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_end = time.time()
            epoch_time = epoch_end - epoch_begin
            remain_time_s = epoch_time * (
                    (train_epoch - epoch) + np.ceil((train_epoch - epoch) / eval_step) * eval_epoch)
            remain_time_h = remain_time_s / 3600
            print(
                f"Epoch : {epoch}, loss : {loss:.4f}, time : {epoch_time:.4f}s, remain time : {remain_time_s:.4f}s ({remain_time_h:.2f}h)",
                end="\r")

            # evaluate
            if (epoch + 1) % eval_step == 0:

                eval_time = time.time()
                p, r, f1, binary_p, binary_r, binary_f1 = self.evaluate(model, eval_epoch, evalN, K, Q, O,
                                                                        train_step=epoch)

                print()
                print(
                    f"Evaluate result of epoch {epoch} - eval time : {time.time() - eval_time:.4f}s, P : {p:.6f}, R : {r:.6f}, F1 : {f1:.6f}")
                if f1 >= best_f1:
                    self.save_model(model, save_ckpt)
                    best_p = p
                    best_r = r
                    best_f1 = f1
                    best_epoch = epoch
                    # test_p, test_r, test_f1, _, _, _ = self.evaluate(model, eval_epoch, evalN, K, Q, O, train_step=epoch, mode="test")
                    print(
                        f"New best performance in epoch {epoch} - P: {best_p:.6f}, R: {best_r:.6f}, F1: {best_f1:.6f}")
                    # print(f"Test performance in epoch {epoch} - P: {test_p:.6f}, R: {test_r:.6f}, F1: {test_f1:.6f}")
                    no_increase_interval = 0
                else:
                    no_increase_interval += 1
                    print(
                        f"Current best performance - P: {best_p:.6f}, R: {best_r:.6f}, F1: {best_f1:.6f} in epoch {best_epoch}")

            # early_stopping
            if no_increase_interval >= self.opt.early_stop_interval:
                print("Early stopped.")
                break

    def draw_embeddings(self, support_attention_mask, support_labels, query_attention_mask, query_labels, query_preds,
                        support_features, query_features, prototype_features, figure_path, id2label, N):
        tsne = TSNE(n_components=2)
        # 对传进来的numpy数组进行各种处理
        if self.opt.use_BIO:
            # 如果是BIO模式，那么需要将i+1和i统一成一个类
            support_labels = (support_labels + 1) // 2
            query_labels = (query_labels + 1) // 2
            prototype_features = prototype_features[
                np.concatenate([np.array([0]), np.arange(1, 2 * self.opt.evalN + 1, 2)])]
        pad_support_labels = support_labels[np.where(support_labels < 0)]
        no_pad_support_labels = support_labels[np.where(support_labels >= 0)]
        pad_support_features = support_features[np.where(support_labels < 0)]
        no_pad_support_features = support_features[np.where(support_labels >= 0)]
        pad_query_labels = query_labels[np.where(query_attention_mask == 0)]
        no_pad_query_labels = query_labels[np.where(query_attention_mask == 1)]
        pad_query_preds = query_preds[np.where(query_attention_mask == 0)]
        no_pad_query_preds = query_preds[np.where(query_attention_mask == 1)]
        pad_query_features = query_features[np.where(query_attention_mask == 0)]
        no_pad_query_features = query_features[np.where(query_attention_mask == 1)]
        support_number = no_pad_support_features.shape[0]
        query_number = no_pad_query_features.shape[0]
        drawing_features = np.concatenate([no_pad_support_features, prototype_features], axis=0)
        tsne_drawing_features = tsne.fit_transform(drawing_features)
        tsne_support_features = tsne_drawing_features[:support_number]
        # tsne_query_features = tsne_drawing_features[support_number:support_number+query_number]
        tsne_prototype_features = tsne_drawing_features[support_number:]
        # print("tsne_query_features: ", tsne_query_features.shape)
        # print("no_pad_query_labels: ", no_pad_query_labels.shape)
        # print("no_pad_query_preds: ", no_pad_query_preds.shape)
        # correct_tsne_query_features = tsne_query_features[np.where(no_pad_query_labels == no_pad_query_preds)]
        # incorrect_tsne_query_features = tsne_query_features[np.where(no_pad_query_labels != no_pad_query_preds)]
        correct_query_preds = no_pad_query_preds[np.where(no_pad_query_labels == no_pad_query_preds)]
        incorrect_query_preds = no_pad_query_preds[np.where(no_pad_query_labels != no_pad_query_preds)]
        prototype_labels = np.array(list(set(no_pad_support_labels.tolist())))
        # print("tsne_support_features: ", tsne_support_features.shape)
        # print("correct_tsne_query_features: ", correct_tsne_query_features.shape)
        # print("incorrect_tsne_query_features: ", incorrect_tsne_query_features.shape)
        # print("tsne_prototype_features: ", tsne_prototype_features.shape)
        # 进行可视化
        v_x = tsne_drawing_features
        v_y = np.concatenate([no_pad_support_labels, no_pad_query_labels, prototype_labels], axis=0)
        cmap = plt.get_cmap('Set1', len(set(v_y.tolist())))  # 数字与颜色的转换
        fig = plt.figure(figsize=(4, 3.5))
        ax = fig.add_subplot(1, 1, 1)
        classes_index = list(range(1, N + 1))
        colors = ["red", "darkorange", "cornflowerblue", "royalblue", "y"]
        for key in classes_index:
            if key == -100 or key == 0:
                continue
            support_index = np.where(no_pad_support_labels == key)
            query_index = np.where(no_pad_query_labels == key)
            correct_query_index = np.where(correct_query_preds == key)
            incorrect_query_index = np.where(incorrect_query_preds == key)
            # print("support_index: ", support_index[0].shape)
            # print("correct_query_index: ", correct_query_index[0].shape)
            # print("incorrect_query_index: ", incorrect_query_index[0].shape)
            """
            ax.scatter(tsne_prototype_features[key][0], tsne_prototype_features[key][1],
                       marker='.', color=cmap(classes_index.index(key)), s=400, alpha=1.0)
            """
            ax.scatter(tsne_support_features[support_index][:, 0], tsne_support_features[support_index][:, 1],
                       marker='.', color=colors[key - 1], label=key, s=120, alpha=1.0)
            """
            ax.scatter(tsne_query_features[query_index][:, 0], tsne_query_features[query_index][:, 1],
                       marker='.', color=cmap(classes_index.index(key)), label=key, s=100, alpha=1.0)
            """
            """
            ax.scatter(correct_tsne_query_features[correct_query_index][:, 0],
                           correct_tsne_query_features[correct_query_index][:, 1], marker='^',
                           color=cmap(classes_index.index(key)))
            ax.scatter(incorrect_tsne_query_features[incorrect_query_index][:, 0],
                           incorrect_tsne_query_features[incorrect_query_index][:, 1], marker='x',
                           color=cmap(classes_index.index(key)))
            """
        labels = [id2label[i].split(".")[-1] for i in range(1, N + 1)]
        print(labels)
        ax.legend(labels=[id2label[i].split(".")[-1] for i in range(1, N + 1)], fontsize=8)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(figure_path, format="svg", bbox_inches="tight")
        plt.close()
