#!/bin/bash
#SBATCH -A research
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/home2/pronoy.patra/Segmango_project/segmango_ssh/results_segformer.txt
#SBATCH --nodelist gnode084
#SBATCH --time=96:00:00
#SBATCH --mail-user=janaksinh.ven@research.iiit.ac.in
#SBATCH --mail-type=ALL
    
source ~/.bashrc

conda activate segmango

python mmsegmentation/tools/train.py mangosense_configs/mango_sense_segformer_512.py
# bash mmsegmentation/tools/dist_train.sh mangosense_configs/mango_sense_segformer_512.py 4
# bash tools/dist_train.sh mangosense_configs/2_mango_sense.py 4
# bash tools/dist_train.sh mangosense_configs/2_mango_sense_1.py 4
# bash tools/dist_train.sh mangosense_configs/2_mango_sense_pspnet.py 4
# bash tools/dist_train.sh mangosense_configs/2_mango_sense_pspnet_1.py 4
# bash tools/dist_train.sh mangosense_configs/2_mango_sense_segformer.py 4
# bash tools/dist_train.sh mangosense_configs/2_mango_sense_segformer_1.py 4
# bash tools/dist_train.sh mangosense_configs/2_mango_sense_swin.py 4
# bash tools/dist_train.sh mangosense_configs/2_mango_sense_swin_1.py 4
# bash tools/dist_train.sh mangosense_configs/2_mango_sense_mask2former.py 4

# python Weakly-Supervised-Learning-Citrus-Pest-Benchmark/train_efficientnet_binary_1200.py

# python train_node_classification_wandb_discrt.py --dataset_name wikipedia --model_name TGN --load_best_configs --num_runs 5 --gpu 0 --wnd_runs 250 --lr_min 3e-5 --lr_max 3e-3 --coverage 0.8 --lambda_val 32