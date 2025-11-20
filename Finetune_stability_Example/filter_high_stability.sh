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


export CUDA_VISIBLE_DEVICES=0

cd /home/acd13855wx/projects/vita/rna2stab/

python filter_high_stability.py