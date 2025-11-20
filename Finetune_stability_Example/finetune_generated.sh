#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=05:00:00
#PBS -P gch51598  
#PBS -o output
#PBS -e log

source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate vita

export CUDA_VISIBLE_DEVICES=0

cd /home/acd13855wx/projects/vita/rna2stab/finetune/

python finetune_generated.py \
  --ckpt /home/acd13855wx/projects/vita/rna2stab/output_finetune_stability_gpt/best_model.pt \
  --outdir /home/acd13855wx/projects/vita/rna2stab/finetune/finetune_generated \
  --num 1000 --min_len 100 --max_len 1000 \
  --temperature 1.0 --top_k 0 --seed 42

