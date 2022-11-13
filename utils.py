# -*- coding: utf-8 -*-
import json
import os
from tqdm import tqdm
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def statistics_of_fewevent(data_dir):
    """
    统计数据集的各种信息
    绘制直方图以表示数据集分布
    :param data_dir:
    :return:
    """
    train_set_path = os.path.join(data_dir, "meta_train_dataset.json")
    dev_set_path = os.path.join(data_dir, "meta_dev_dataset.json")
    test_set_path = os.path.join(data_dir, "meta_test_dataset.json")
    all_samples = []
    all_length = []
    max_len = 0
    sum_len = 0
    multi_token_trigger_num = 0
    all_trigger_num = 0
    sent_longer_than_128 = 0
    sent_longer_than_64 = 0
    statistics = {
        "max_len": 0,
        "avg_len": 0,
        "multi_token_trigger_percentage": 0.0,
        "sent_longer_than_128_percentage": 0.0,
        "sent_longer_than_64_percentage": 0.0,
        "trigger_num": 0,
        "sent_num": 0
    }
    for file_path in [train_set_path, dev_set_path, test_set_path]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for event_type, event_instances in data.items():
                all_samples.extend(event_instances)
                for sample in event_instances:
                    if len(sample["tokens"]) <= 9:
                        print(sample["tokens"], sample["trigger"], event_type)
    for sample in tqdm(all_samples):
        if len(sample["tokens"]) < 128:
            all_length.append(len(sample["tokens"]))
        if len(sample["tokens"]) > max_len:
            max_len = len(sample["tokens"])
        if len(sample["tokens"]) > 128:
            sent_longer_than_128 += 1
        if len(sample["tokens"]) > 64:
            sent_longer_than_64 += 1
        sum_len += len(sample["tokens"])
        if len(sample["trigger"]) > 1:
            multi_token_trigger_num += 1
        all_trigger_num += len(sample["trigger"])
    statistics["max_len"] = max_len
    statistics["avg_len"] = sum_len / len(all_samples)
    statistics["multi_token_trigger_percentage"] = multi_token_trigger_num / all_trigger_num
    statistics["sent_longer_than_128_percentage"] = sent_longer_than_128 / len(all_samples)
    statistics["sent_longer_than_64_percentage"] = sent_longer_than_64 / len(all_samples)
    statistics["trigger_num"] = all_trigger_num
    statistics["sent_num"] = len(all_samples)
    # 绘制直方图，统计句子长度的分布情况
    all_length = np.array(all_length, dtype=np.int64)
    fig, ax = plt.subplots(1, 1)
    sns.distplot(all_length, kde=False)
    plt.xlabel("length")
    plt.ylabel("Number")
    plt.savefig("img/length_distribution.svg", format="svg", bbox_inches="tight")
    plt.show()
    plt.close()
    return statistics


def t_test(x1, x2):
    t_score, p_value = ttest_ind(x1, x2, equal_var=True)
    return {"t_score": t_score, "p_value": p_value}


if __name__ == "__main__":
    statistics = statistics_of_fewevent("data/FewEvent")
    print(statistics)
    our_scores = np.array([64.88, 61.31, 63.48, 61.97, 65.84])
    our_v2_scores = np.array([65.59, 65.30, 64.32, 64.59, 65.34])
    print(t_test(our_scores, our_v2_scores))
