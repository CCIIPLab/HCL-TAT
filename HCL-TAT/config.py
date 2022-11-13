# -*- coding: utf-8 -*-

seed = None

dataset = "fewevent"
encoder_path = '/home/xxx/data/huggingface/bert-base-uncased'

# model settings 
model = "pa_crf"
distance_metric = "dot"  # [dot, cos, euclidean]
sample_num = 5

# encoder settings
encoder = "bert"
feature_size = 768
max_length = 128

trainN = 5
evalN = 5
K = 5
Q = 1
O = 0   # 如果O为0，则使用Other类的embedding计算prototype，否则使用O个随机初始化向量来表示（多实例模式）
use_BIO = False  # 是否使用BIO模式
alpha = 0.5
beta = 0.5

batch_size = 1
num_workers = 8

dropout = 0.1
optimizer = "adamw"
learning_rate = 1e-5
warmup_step = 100
scheduler_step = 1000
temperature_alpha = 0.5
temperature_beta = 0.1
threshold = "None"    # 可以取None，mean或者max

train_epoch = 20000
eval_epoch = 1000
eval_step = 500
test_epoch = 3000
early_stop_interval = 5    # 如果超过一定验证次数指标没有提升，就早停以节约时间。默认值是5

ckpt_dir = "checkpoint/"
load_ckpt = None
save_ckpt = None
embedding_visualization_path = None
result_dir = None

device = None
test=False

notes=""
contrastive = "None"    # [None, Normal, Weighted]
