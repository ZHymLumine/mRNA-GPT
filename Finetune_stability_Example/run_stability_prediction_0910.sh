#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=03:00:00
#PBS -P gch51598  
#PBS -o output
#PBS -e log

source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
source ~/.bashrc
conda activate vita

export CUDA_VISIBLE_DEVICES=0

cd /home/acd13855wx/projects/vita/rna2stab/

echo "🔍 开始预测生成序列的稳定性值..."
echo "开始时间: $(date)"

python predict_stability_generated_v3.py \
    --model /home/acd13855wx/projects/vita/rna2stab/best_transformer_model.pth \
    --finetuned_fasta /home/acd13855wx/projects/vita/rna2stab/finetune/finetune_generated/finetuned_generated.fasta \
    --pretrained_fasta /home/acd13855wx/projects/vita/rna2stab/finetune/pretrained_generated/pretrained_generated.fasta \
    --output /home/acd13855wx/projects/vita/rna2stab/stability_comparison0910 \
    --batch_size 16 \
    --finetuned_label "Finetuned" \
    --pretrained_label "Pretrained"

echo "预测完成时间: $(date)"
echo "📁 结果保存在 stability_comparison 文件夹中"
echo "📊 可以查看生成的图表和统计报告"
