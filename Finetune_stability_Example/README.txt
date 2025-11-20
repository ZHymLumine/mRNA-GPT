1. Train the rna2stability model

input:mRNA_Stability.csv

rna2sta.py
rna2sta.sh

Output model:best_transformer_model.pth

2. Select the True-Positive data

Input file: mRNA_Stability.csv
filter_high_stability.py
filter_high_stability.sh

Output file:high_stability_sequences.csv


3. Supervised finetuning process

Input:pretrained_archaea_ckpt_62000.pt
finetune_stability_archaea.py
run_stability_finetune.sh

Output finetuned model: best_model.pt


4. mRNA generation by finetuned model and pretrained model


Finetuned generated:
finetune_generated.py
finetune_generated.sh

Pretrained generated:
generate_sequences_pretrained_matchlen.py
run_pretrained_generation_0910.sh


Output file:finetuned_generated.fasta,pretrained_generated.fasta


5. Comparison of finetuned generated seq and pretrained generated seq

Input file: finetuned_generated.fasta,pretrained_generated.fasta
predict_stability_generated_v3.py
run_stability_prediction_0910.sh


