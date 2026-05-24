#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/home2/pronoy.patra/Segmango_project/segmango_ssh/results_segmango_per_tree.txt
#SBATCH --nodelist gnode077
#SBATCH --time=96:00:00
#SBATCH --mail-user=janaksinh.ven@research.iiit.ac.in
#SBATCH --mail-type=ALL
    
source ~/.bashrc

conda activate segmango
python train_segmango_per_tree.py --fold 1 --variant b1 --unfreez_epoch 50
python train_segmango_per_tree.py --weather --fold 1 --variant b1 --unfreez_epoch 50
python train_segmango_per_tree.py --weather --scale --fold 1 --variant b1 --unfreez_epoch 50