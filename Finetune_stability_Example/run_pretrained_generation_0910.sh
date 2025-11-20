#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -P gch51598  
#PBS -o output
#PBS -e log

source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate vita

export CUDA_VISIBLE_DEVICES=0

cd /home/acd13855wx/projects/vita/rna2stab/finetune/

echo "🔍 使用预训练模型 ckpt_62000.pt 生成1000条序列..."
echo "开始时间: $(date)"

python generate_sequences_pretrained_matchlen.py \
    --ckpt /home/acd13855wx/projects/vita/rna2stab/finetune/ckpt_62000.pt \
    --vocab /home/acd13855wx/projects/vita/rna2stab/finetune/vocab.txt \
    --outdir /home/acd13855wx/projects/vita/rna2stab/finetune/pretrained_generated \
    --ref_fasta /home/acd13855wx/projects/vita/rna2stab/finetune/finetune_generated/finetuned_generated.fasta \
    --num 1000 \
    --temperature 1.0 \
    --top_k 0 \
    --seed 42

echo "生成完成时间: $(date)"
echo "📁 结果保存在 pretrained_generated 文件夹中"
