#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd
from rna2te import RNATranslationEfficiencyPredictor

def predict_te_from_sequences(sequences, model_path="best_transformer_model.pth"):

    predictor = RNATranslationEfficiencyPredictor(model_path)
    predictor.preprocess_data("ecoli_TE_CDS_final.csv")
    # 预测翻译效率
    predictions = predictor.predict(sequences)
    
    return predictions

def predict_te_from_file(input_file, output_file=None, model_path="best_transformer_model.pth"):
    """
    从文件读取RNA序列并预测翻译效率，然后保存结果
    
    参数:
    input_file: str, 输入文件路径，CSV格式，包含一列名为'sequence'的RNA序列
    output_file: str, 输出文件路径，默认为input_file添加_predictions后缀
    model_path: str, 模型路径
    
    返回:
    df_result: DataFrame, 包含序列和预测结果
    """
    # 读取序列
    df = pd.read_csv(input_file)
    
    if 'sequence' not in df.columns:
        raise ValueError("输入文件必须包含'sequence'列")
    
    sequences = df['sequence'].tolist()
    
    # 预测翻译效率
    predictions = predict_te_from_sequences(sequences, model_path)
    
    # 创建结果DataFrame
    df_result = df.copy()
    df_result['predicted_te'] = predictions
    
    # 保存结果
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_predictions.csv"
    
    df_result.to_csv(output_file, index=False)
    print(f"预测结果已保存至: {output_file}")
    
    return df_result

def main():
    parser = argparse.ArgumentParser(description='预测RNA序列的翻译效率')
    parser.add_argument('--input', '-i', type=str, help='输入文件路径，CSV格式，包含一列名为sequence的RNA序列')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出文件路径')
    parser.add_argument('--model', '-m', type=str, default="best_transformer_model.pth", help='模型路径')
    parser.add_argument('--sequences', '-s', nargs='+', help='直接输入RNA序列列表进行预测')
    
    args = parser.parse_args()
    
    if args.input:
        predict_te_from_file(args.input, args.output, args.model)
    elif args.sequences:
        predictions = predict_te_from_sequences(args.sequences, args.model)
        for seq, pred in zip(args.sequences, predictions):
            print(f"序列: {seq[:30]}{'...' if len(seq) > 30 else ''}")
            print(f"预测翻译效率: {pred:.4f}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 