#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=01:30:00
#PBS -P gch51598  
#PBS -o output
#PBS -e log

source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate vita

#export WANDB_API_KEY="d5c4f0b14e5e8ad8fad5ca6a880fc6a7466a9f84"

export CUDA_VISIBLE_DEVICES=0

cd /home/acd13855wx/projects/vita/rna2stab/

python rna2sta.py