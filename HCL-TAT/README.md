This code is for paper "HCL-TAT: A Hybrid Contrastive Learning Method for Few-shot Event Detection with Task-adaptive Threshold" accepted to Findings of EMNLP 2022.

Requirements
---

```
python=3.7
pytorch=1.10.2
cuda=10.2
transformers=2.8.0
```

Usage
---

The FewEvent dataset is under "data" dir. First, you should **modify line 165-170 in main.py, line 32 in framework.py and line 226 in dataloader.py**, by replacing the paths to yours. Then, you can train and evaluate the model by running the two following scripts.

1. Training

```shell
bash run.sh
```

You can modify the parameters in run.sh to train the model in different settings.

The checkpoint will be saved under the "checkpoint" dir. And if you specify the **"embedding_visualization_path"** and **"result_dir"**, the visualization and metric results will be saved as well.

For example, if you want to train our HCL-TAT in 5-way-5-shot setting,  the content in run.sh should be like:

```shell
for ((i=1; i<=5; i+=1))
do
  python main.py\
   --model=proto_dot\
   --trainN=5\
   --evalN=5\
   --K=5\
   --Q=1\
   --O=0\
   --distance_metric="proto"\
   --contrastive="Normal"\
   --temperature_alpha=0.5\
   --temperature_beta=0.1\
   --alpha=0.5\
   --beta=0.5\
   --threshold="mean"\
   --result_dir="result/proto_hcl";
done
```

The value of O and P should keep 0 and 1 respectively.  

After each iteration, the code will print the results on test set in the screen, and finally, you can get 5 checkpoints. Note that results on dev set are lower than results on test set, which is normal in this split.

2. Testing

```shell
bash evaluate.sh
```

Similar to training stage, you can modify the parameters to evaluate models in different settings. If you want to evaluate a 5-way-5-shot HCL-TAT model, then the content in evaluate.sh should be like:

```shell
  python main.py\
   --model=proto_dot\
   --trainN=5\
   --evalN=5\
   --K=5\
   --Q=1\
   --O=0\
   --distance_metric="proto"\
   --contrastive="Normal"\
   --temperature_alpha=0.5\
   --temperature_beta=0.1\
   --alpha=0.5\
   --beta=0.5\
   --threshold="mean"\
   --test\
   --load_ckpt="proto_dot_fewevent_5_5_xxxxxxxx"\
   --embedding_visualization_path="embedding_visualization/proto_hcl"\
```

Please specify the **"load_checkpoint"** parameter to load the model.
