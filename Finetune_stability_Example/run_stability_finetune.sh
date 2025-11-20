#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=05:30:00
#PBS -P gch51598  
#PBS -o output
#PBS -e log

source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate vita


export CUDA_VISIBLE_DEVICES=0

cd /home/acd13855wx/projects/vita/rna2stab/finetune

# 创建日志目录
mkdir -p log

echo "开始高稳定性mRNA序列微调..."
echo "使用数据集: high_stability_sequences.csv"
echo "开始时间: $(date)"

# 运行微调脚本
python finetune_stability_archaea.py \
    --csv /home/acd13855wx/projects/vita/rna2stab/high_stability_sequences.csv \
    --vocab /home/acd13855wx/projects/vita/rna2stab/finetune/vocab.txt \
    --ckpt /home/acd13855wx/projects/vita/rna2stab/finetune/ckpt_62000.pt \
    --outdir /home/acd13855wx/projects/vita/rna2stab/output_finetune_stability_gpt \
    --stability_threshold 0.5 \
    --epochs 300 \
    --batch_size 16 \
    --lr 1e-4 \
    --val_size 0.15

echo "微调完成时间: $(date)"
echo "输出目录: /home/acd13855wx/projects/vita/rna2stab/output_finetune_stability_gpt"
