# -*- coding: utf-8 -*-

import os
import time
import argparse

import random
import numpy as np
import nni

import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from transformers import AdamW, get_linear_schedule_with_warmup

from model import Proto, Relation, Match, ProtoDot, PACRF

from model.encoder import BertEncoder, RobertaEncoder
from dataloader import get_loader
from framework import Framework
from metric import Metric
import config


def main():
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=config.seed, type=int,
                        help='seed')

    parser.add_argument('--dataset', default=config.dataset, type=str,
                        help='fewevent')

    parser.add_argument('--model', default=config.model, type=str,
                        help='model')
    parser.add_argument('--distance_metric', default=config.distance_metric, type=str,
                        help='dot or cos or euclidean')

    parser.add_argument('--sample_num', default=config.sample_num, type=int,
                        help='sample num of MC')

    parser.add_argument('--encoder', default=config.encoder, type=str,
                        help='bert')
    parser.add_argument('--feature_size', default=config.feature_size, type=int,
                        help='feature size')
    parser.add_argument('--max_length', default=config.max_length, type=int,
                        help='max sentence length')
    parser.add_argument('--encoder_path', default=config.encoder_path, type=str,
                        help='pretrained encoder path')

    parser.add_argument('--trainN', default=config.trainN, type=int,
                        help='train N')
    parser.add_argument('--evalN', default=config.evalN, type=int,
                        help='eval N')
    parser.add_argument('--K', default=config.K, type=int,
                        help="K")
    parser.add_argument('--Q', default=config.Q, type=int,
                        help="Q")
    parser.add_argument('--O', default=config.O, type=int,
                        help="O")
    parser.add_argument("--use_BIO", action="store_true",
                        help="whether to use BIO tag scheme")
    parser.add_argument('--alpha', default=config.alpha, type=float,
                        help="support set内部对比学习loss比例")
    parser.add_argument('--beta', default=config.beta, type=float,
                        help="query和prototype间对比学习loss比例")

    parser.add_argument('--batch_size', default=config.batch_size, type=int,
                        help='batch size')
    parser.add_argument('--num_workers', default=config.num_workers, type=int,
                        help='number of worker in dataloader')

    parser.add_argument('--dropout', default=config.dropout, type=float,
                        help='dropout rate')
    parser.add_argument('--optimizer', default=config.optimizer, type=str,
                        help='sgd or adam or adamw')
    parser.add_argument('--learning_rate', default=config.learning_rate, type=float,
                        help='learnint rate')
    parser.add_argument('--warmup_step', default=config.warmup_step, type=int,
                        help='warmup step of bert')
    parser.add_argument('--scheduler_step', default=config.scheduler_step, type=int,
                        help='scheduler step')

    parser.add_argument('--train_epoch', default=config.train_epoch, type=int,
                        help='train epoch')
    parser.add_argument('--eval_epoch', default=config.eval_epoch, type=int,
                        help='eval epoch')
    parser.add_argument('--eval_step', default=config.eval_step, type=int,
                        help='eval step')
    parser.add_argument('--test_epoch', default=config.test_epoch, type=int,
                        help='test epoch')
    parser.add_argument("--early_stop_interval", default=config.early_stop_interval, type=int,
                        help="early stop")

    parser.add_argument('--ckpt_dir', default=config.ckpt_dir, type=str,
                        help='checkpoint dir')
    parser.add_argument('--embedding_visualization_path', default=config.embedding_visualization_path, type=str,
                        help='dir to store embedding visualization results')
    parser.add_argument('--load_ckpt', default=config.load_ckpt, type=str,
                        help='load checkpoint')
    parser.add_argument('--save_ckpt', default=config.save_ckpt, type=str,
                        help='save checkpoint')
    parser.add_argument('--result_dir', default=config.result_dir, type=str,
                        help='result_dir')

    parser.add_argument('--device', default=config.device, type=str,
                        help='device')
    parser.add_argument('--test', default=config.test, action="store_true",
                        help='test mode')
    parser.add_argument('--random_split', action="store_true",
                        help='whether to use random split')
    parser.add_argument('--contrastive', default="Normal", type=str,
                        help='mode of contrastive learning')
    parser.add_argument('--temperature_alpha', default=config.temperature_alpha, type=float,
                        help='scaler parameter in contrastive loss')
    parser.add_argument('--temperature_beta', default=config.temperature_beta, type=float,
                        help='scaler parameter in contrastive loss')
    parser.add_argument('--threshold', default="None", type=str,
                        help='mode of threshold')

    parser.add_argument('--notes', default=config.notes, type=str,
                        help='experiment notes')

    opt = parser.parse_args()

    # experiment notes
    print("Experiment notes :", opt.notes)

    # set seed
    if opt.seed is None:
        opt.seed = round((time.time() * 1e4) % 1e4)
    print(f"Seed: {opt.seed}")
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    timestrap = time.strftime('%Y%m%d_%H%M%S')
    if opt.save_ckpt is None:
        opt.save_ckpt = os.path.join(opt.ckpt_dir,
                                     "_".join([opt.model,
                                               opt.dataset,
                                               str(opt.evalN),
                                               str(opt.K),
                                               timestrap + ".ckpt"]))
        print(f"Save checkpoint : {opt.save_ckpt}")
    else:
        opt.save_ckpt = os.path.join(opt.ckpt_dir, opt.save_ckpt + ".ckpt")

    if opt.load_ckpt is not None:
        opt.load_ckpt = os.path.join(opt.ckpt_dir, opt.load_ckpt + ".ckpt")

    if opt.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(opt.device)

    print("Hyperparameters :", opt)

    # define encoder
    if opt.encoder == "bert":
        opt.encoder_path = "/home/xxx/data/huggingface/bert-base-uncased"
        encoder = BertEncoder(opt.encoder_path, opt.max_length)
    else:
        opt.encoder_path = "/home/xxx/data/huggingface/roberta-base"
        encoder = RobertaEncoder(opt.encoder_path, opt.max_length)

    # load dataset    
    train_dataset = get_loader(opt.dataset,
                               "TRAIN",
                               opt.max_length,
                               encoder.tokenize,
                               opt.trainN, opt.K, opt.Q, opt.O,
                               opt.batch_size, use_BIO=opt.use_BIO)
    dev_dataset = get_loader(opt.dataset,
                             "DEV",
                             opt.max_length,
                             encoder.tokenize,
                             opt.evalN, opt.K, opt.Q, opt.O,
                             opt.batch_size, use_BIO=opt.use_BIO)
    test_dataset = get_loader(opt.dataset,
                              "TEST",
                              opt.max_length,
                              encoder.tokenize,
                              opt.evalN, opt.K, opt.Q, opt.O,
                              opt.batch_size, use_BIO=opt.use_BIO)

    # define model
    if opt.model == "proto":
        model = Proto(encoder, opt.feature_size, opt.max_length, opt.dropout)
    elif opt.model == "match":
        model = Match(encoder, opt.feature_size, opt.max_length, opt.dropout)
    elif opt.model == "relation":
        model = Relation(encoder, opt.feature_size, opt.max_length, opt.dropout)
    elif opt.model == "proto_dot":
        model = ProtoDot(encoder, opt)
    elif opt.model == "pa_crf":
        model = PACRF(opt.evalN, opt.sample_num, encoder, opt.feature_size, opt.max_length, opt.dropout)
    else:
        raise Exception("Invalid model!")
    model.to(device)

    # define optimizer and scheduler    
    if opt.optimizer == "adamw":
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(parameters_to_optimize, lr=opt.learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opt.warmup_step,
                                                    num_training_steps=opt.train_epoch)
    elif opt.optimizer == "sgd":
        parameters_to_optimize = list(model.parameters())
        optimizer = SGD(parameters_to_optimize, lr=opt.learning_rate)
        scheduler = StepLR(optimizer, opt.scheduler_step)
    elif opt.optimizer == "adam":
        parameters_to_optimize = list(model.parameters())
        optimizer = Adam(parameters_to_optimize, lr=opt.learning_rate)
        scheduler = StepLR(optimizer, opt.scheduler_step)
    else:
        raise ValueError("Invalid optimizer")

    # define metric
    metric = Metric()

    # define framework
    framework = Framework(train_dataset=train_dataset,
                          dev_dataset=dev_dataset,
                          test_dataset=test_dataset,
                          metric=metric,
                          device=device,
                          opt=opt)
    # train
    if not opt.test:
        framework.train(model,
                        opt.trainN, opt.evalN, opt.K, opt.Q, opt.O,
                        optimizer,
                        scheduler,
                        opt.train_epoch,
                        opt.eval_epoch,
                        opt.eval_step,
                        load_ckpt=opt.load_ckpt,
                        save_ckpt=opt.save_ckpt)
        checkpoint = opt.save_ckpt
    else:
        checkpoint = opt.load_ckpt

    # test
    P, R, F1, binary_p, binary_r, binary_f1 = framework.evaluate(model,
                                                                                    opt.test_epoch,
                                                                                    opt.evalN, opt.K, opt.Q, opt.O,
                                                                                    train_step=opt.train_epoch - 1,
                                                                                    mode="test",
                                                                                    load_ckpt=checkpoint)
    print(
        f"Test result - P : {P:.6f}, R : {R:.6f}, F1 : {F1:.6f}, 二分类P : {binary_p:.6f}, 二分类R : {binary_r:.6f}, 二分类F1 : {binary_f1:.6f}")

    # finish and write result
    print("Hyperparameters :", opt)
    print("Experiment notes :", opt.notes)
    result_dir = opt.result_dir
    if result_dir is not None:
        if opt.contrastive != "None":
            result_file = os.path.join(result_dir, timestrap + "_" + str(opt.trainN) + "way_" + str(
                opt.K) + "shot_" + opt.distance_metric + "_O=" + str(opt.O) + "_" + opt.contrastive + ".log")
        else:
            result_file = os.path.join(result_dir, timestrap + "_" + str(opt.trainN) + "way_" + str(
                opt.K) + "shot_" + "O=" + str(opt.O) + ".log")
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(str(opt) + "\n")
            f.write("Precision: " + str(P) + "\n")
            f.write("Recall: " + str(R) + "\n")
            f.write("F1: " + str(F1) + "\n")
            f.write("二分类Precision: " + str(binary_p) + "\n")
            f.write("二分类Recall: " + str(binary_r) + "\n")
            f.write("二分类F1: " + str(binary_f1) + "\n")


if __name__ == "__main__":
    main()
