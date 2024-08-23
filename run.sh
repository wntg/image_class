gpus="0 1 2 3 4 5 6 7"
num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
torchrun --nproc_per_node=$num_gpu  train.py --gpu $gpus
