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