# -*- coding: utf-8 -*-

import os
import json
import random
from collections import OrderedDict, Collection
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class BaseDataset(Dataset):
    def __init__(self, 
                 dataset_path,
                 max_length, 
                 tokenize,
                 N, K, Q, O, use_BIO=True):
        self.raw_data = json.load(open(dataset_path, "r"))
        self.classes = self.raw_data.keys()
        
        self.max_length = max_length - 2
        self.tokenize = tokenize
        
        self.N = N
        self.K = K
        self.Q = Q
        self.O = O
        self.use_BIO = use_BIO
        
    def __len__(self):
        return 99999999
    
    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        label2id, id2label = self.build_dict(target_classes)
        
        support_set = {'tokens': [], 'trigger_label': [], 'B-mask': [], 'I-mask': [], 'att-mask': [], 'text-mask': []}
        query_set = {'tokens': [], 'trigger_label': [], 'B-mask': [], 'I-mask': [], 'att-mask': [], 'text-mask': []}
        
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.raw_data[class_name]))), 
                    self.K + self.Q, False)
            
            count = 0
            for j in indices:
                if count < self.K:
                    instance = self.preprocess(self.raw_data[class_name][j], [class_name])
                    token_ids, label_ids, B_mask, I_mask, att_mask, text_mask = self.tokenize(instance, label2id)
                    
                    support_set['tokens'].append(token_ids)
                    support_set['trigger_label'].append(label_ids)
                    support_set['B-mask'].append(B_mask)
                    support_set['I-mask'].append(I_mask)
                    support_set['att-mask'].append(att_mask)
                    support_set['text-mask'].append(text_mask)
                else:
                    instance = self.preprocess(self.raw_data[class_name][j], target_classes)
                    token_ids, label_ids, B_mask, I_mask, att_mask, text_mask = self.tokenize(instance, label2id)
                    
                    query_set['tokens'].append(token_ids)
                    query_set['trigger_label'].append(label_ids)
                    query_set['B-mask'].append(B_mask)
                    query_set['I-mask'].append(I_mask)
                    query_set['att-mask'].append(att_mask)
                    query_set['text-mask'].append(text_mask)
                count += 1
        
        for k, v in support_set.items():
            support_set[k] = torch.stack(v)
        
        for k, v in query_set.items():
            query_set[k] = torch.stack(v)
        
        return support_set, query_set, id2label
    
    def preprocess(self, instance, event_type_list):
        raise NotImplementedError
    
    def build_dict(self, event_type_list):
        label2id = OrderedDict()
        id2label = OrderedDict()

        label2id['O'] = 0
        id2label[0] = 'O'
        label2id['PAD'] = -100
        id2label[-100] = 'PAD'
        
        for i, event_type in enumerate(event_type_list):
            if self.use_BIO:
                label2id['B-' + event_type] = 2*i + 1
                label2id['I-' + event_type] = 2*i + 2
                id2label[2*i + 1] = 'B-' + event_type
                id2label[2*i + 2] = 'I-' + event_type
            else:
                label2id['I-' + event_type] = i+1
                id2label[i+1] = 'I-' + event_type
        
        return label2id, id2label
            
class FewEventDataset(BaseDataset):

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        label2id, id2label = self.build_dict(target_classes)
        
        support_set = {'tokens': [], 'trigger_label': [], 'B-mask': [], 'I-mask': [], "att-mask": [], 'text-mask': []}
        query_set = {'tokens': [], 'trigger_label': [], 'B-mask': [], 'I-mask': [], "att-mask": [], 'text-mask': []}
        
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.raw_data[class_name]))), 
                    self.K + self.Q, False)
            
            count = 0
            for j in indices:
                if count < self.K:
                    instance = self.preprocess(self.raw_data[class_name][j], class_name, [class_name])
                    token_ids, label_ids, B_mask, I_mask, att_mask, text_mask = self.tokenize(instance, label2id)
                    
                    support_set['tokens'].append(token_ids)
                    support_set['trigger_label'].append(label_ids)
                    support_set['B-mask'].append(B_mask)
                    support_set['I-mask'].append(I_mask)
                    support_set['att-mask'].append(att_mask)
                    support_set['text-mask'].append(text_mask)
                else:
                    instance = self.preprocess(self.raw_data[class_name][j], class_name, target_classes)
                    token_ids, label_ids, B_mask, I_mask, att_mask, text_mask = self.tokenize(instance, label2id)
                    
                    query_set['tokens'].append(token_ids)
                    query_set['trigger_label'].append(label_ids)
                    query_set['B-mask'].append(B_mask)
                    query_set['I-mask'].append(I_mask)
                    query_set['att-mask'].append(att_mask)
                    query_set['text-mask'].append(text_mask)
                count += 1
        
        for k, v in support_set.items():
            support_set[k] = torch.stack(v)
        
        for k, v in query_set.items():
            query_set[k] = torch.stack(v)
        
        return support_set, query_set, id2label


    def preprocess(self, instance, event_type, event_type_list):
        result = {'tokens': [], 'trigger_label': [], 'B-mask':[], 'I-mask': []}

        sentence = instance['tokens']
        result['tokens'] = sentence
        
        trigger_label = ['O'] * len(sentence)
        B_mask = [0] * len(sentence)
        I_mask = [0] * len(sentence)
        
        trigger_length = len(instance['trigger'])
        trigger_start_pos = instance['position'][0]
        trigger_end_pos = trigger_start_pos + trigger_length
        for i in range(trigger_start_pos, trigger_end_pos):
            if self.use_BIO:
                if i == trigger_start_pos:
                    trigger_label[i] = f"B-{event_type}"
                    B_mask[i] = 1
                else:
                    trigger_label[i] = f"I-{event_type}"
                    I_mask[i] = 1
            else:
                trigger_label[i] = f"I-{event_type}"
                I_mask[i] = 1
        result['trigger_label'] = trigger_label
        result['B-mask'] = B_mask
        result['I-mask'] = I_mask
        
        return result

def collate_fn(data):
    batch_support = {'tokens': [], 'trigger_label': [], 'B-mask':[], 'I-mask': [], 'att-mask': [], 'text-mask': []}
    batch_query = {'tokens': [], 'trigger_label': [], 'B-mask':[], 'I-mask': [], 'att-mask': [], 'text-mask': []}
    batch_id2label = []
    
    support_sets, query_sets, id2labels = zip(*data)
    
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k].append(support_sets[i][k])
        for k in query_sets[i]:
            batch_query[k].append(query_sets[i][k])
        batch_id2label.append(id2labels[i])
    
    for k in batch_support:
        batch_support[k] = torch.cat(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.cat(batch_query[k], 0)
    
    return batch_support, batch_query, batch_id2label


def split_json_data(json_data: dict):
    train_data, dev_data, test_data = {}, {}, {}
    event_types = list(json_data.keys())
    random.shuffle(event_types)
    train_types = event_types[:80]
    dev_types = event_types[80: 90]
    test_types = event_types[90: 100]
    for k in train_types:
        train_data[k] = json_data[k]
    for k in dev_types:
        dev_data[k] = json_data[k]
    for k in test_types:
        test_data[k] = json_data[k]
    return train_data, dev_data, test_data


def get_loader(dataset_name,
               mode,
               max_length,
               tokenize,
               N, K, Q, O,
               batch_size,
               use_BIO=False,
               num_workers=4,
               collate_fn=collate_fn):
    root_data_dir = "/home/xxx/code/PA-CRF/data"

    if mode == "TRAIN":
        data_file = "meta_train_dataset.json"
    elif mode == "DEV":
        data_file = "meta_dev_dataset.json"
    elif mode == "TEST":
        data_file = "meta_test_dataset.json"
    else:
        raise ValueError("Error mode!")

    dataset_path = os.path.join(root_data_dir, "FewEvent", data_file)
    dataset = FewEventDataset(dataset_path,
                              max_length,
                              tokenize,
                              N, K, Q, O, use_BIO)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
    return iter(dataloader)

