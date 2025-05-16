#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量预测脚本：处理大量RNA序列并预测翻译效率
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from rna2te import RNATranslationEfficiencyPredictor

def batch_predict(input_file, output_file, model_path="best_transformer_model.pth", batch_size=64):
    """
    批量预测RNA序列的翻译效率
    
    参数:
    input_file: str, 输入文件路径，CSV或FASTA格式
    output_file: str, 输出文件路径
    model_path: str, 模型路径
    batch_size: int, 批处理大小
    """
    # 初始化预测器
    predictor = RNATranslationEfficiencyPredictor(model_path)
    
    # 判断文件格式并读取序列
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
        if 'sequence' not in df.columns:
            raise ValueError("CSV文件必须包含'sequence'列")
        sequences = df['sequence'].tolist()
        ids = df.get('id', [f"seq_{i}" for i in range(len(sequences))])
    elif input_file.endswith(('.fa', '.fasta')):
        sequences = []
        ids = []
        current_id = ""
        current_seq = ""
        
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                        ids.append(current_id)
                    current_id = line[1:].split()[0]
                    current_seq = ""
                else:
                    current_seq += line
            
            if current_seq:  # 添加最后一个序列
                sequences.append(current_seq)
                ids.append(current_id)
    else:
        raise ValueError("不支持的文件格式，请使用CSV或FASTA格式")
    
    # 批量预测
    all_predictions = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="预测中"):
        batch_sequences = sequences[i:i+batch_size]
        batch_predictions = predictor.predict(batch_sequences)
        all_predictions.extend(batch_predictions)
    
    # 保存结果
    results = {
        'id': ids,
        'sequence': sequences,
        'predicted_te': all_predictions
    }
    
    df_result = pd.DataFrame(results)
    df_result.to_csv(output_file, index=False)
    print(f"预测结果已保存至: {output_file}")
    
    # 输出统计信息
    print("\n预测统计:")
    print(f"序列数量: {len(sequences)}")
    print(f"平均翻译效率: {np.mean(all_predictions):.4f}")
    print(f"最大翻译效率: {np.max(all_predictions):.4f}")
    print(f"最小翻译效率: {np.min(all_predictions):.4f}")
    print(f"标准差: {np.std(all_predictions):.4f}")
    
    # 找出翻译效率最高和最低的序列
    max_idx = np.argmax(all_predictions)
    min_idx = np.argmin(all_predictions)
    
    print(f"\n翻译效率最高的序列 (ID: {ids[max_idx]}):")
    print(f"序列: {sequences[max_idx][:50]}..." if len(sequences[max_idx]) > 50 else sequences[max_idx])
    print(f"翻译效率: {all_predictions[max_idx]:.4f}")
    
    print(f"\n翻译效率最低的序列 (ID: {ids[min_idx]}):")
    print(f"序列: {sequences[min_idx][:50]}..." if len(sequences[min_idx]) > 50 else sequences[min_idx])
    print(f"翻译效率: {all_predictions[min_idx]:.4f}")
    
    return df_result

def main():
    parser = argparse.ArgumentParser(description='批量预测RNA序列的翻译效率')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入文件路径，CSV或FASTA格式')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出文件路径')
    parser.add_argument('--model', '-m', type=str, default="best_transformer_model.pth", help='模型路径')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='批处理大小')
    
    args = parser.parse_args()
    
    # 如果未指定输出文件，则使用默认名称
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_predictions.csv"
    
    batch_predict(args.input, args.output, args.model, args.batch_size)

if __name__ == "__main__":
    main() 