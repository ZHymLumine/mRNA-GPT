#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例脚本：展示如何使用RNA翻译效率预测模型
"""

from rna2te import RNATranslationEfficiencyPredictor

def example_single_sequence():
    """单个序列预测示例"""
    # 初始化预测器
    predictor = RNATranslationEfficiencyPredictor("best_transformer_model.pth")
    
    # 预测单个序列
    sequence = "ATGGCGTACGACTACGACTGCGACTACGACGACTGA"
    prediction = predictor.predict([sequence])[0]
    
    print(f"序列: {sequence}")
    print(f"预测翻译效率: {prediction:.4f}")
    
    return prediction

def example_multiple_sequences():
    """多个序列预测示例"""
    # 初始化预测器
    predictor = RNATranslationEfficiencyPredictor("best_transformer_model.pth")
    
    # 预测多个序列
    sequences = [
        "ATGGCGTACGACTACGACTGCGACTACGACGACTGA",
        "ATGCGTCGTATCGATCGATCGATCGATCGATCGTGA",
        "ATGTCGATCGATCGATCGATCGATCGATCGATCTAA"
    ]
    
    predictions = predictor.predict(sequences)
    
    print("\n多序列预测结果:")
    for seq, pred in zip(sequences, predictions):
        print(f"序列: {seq[:20]}...")
        print(f"预测翻译效率: {pred:.4f}\n")
    
    return predictions

def example_compare_sequences():
    """比较不同序列的翻译效率"""
    # 初始化预测器
    predictor = RNATranslationEfficiencyPredictor("best_transformer_model.pth")
    
    # 原始序列
    original_seq = "ATGGCGTACGACTACGACTGCGACTACGACGACTGA"
    
    # 变异序列（模拟密码子优化或突变）
    variant_seqs = [
        "ATGGCGTATGATTACGACTGCGACTACGACGACTGA",  # 单个密码子变化
        "ATGGCGTACGACTACGACTGCGATTACGACGACTGA",  # 另一个变化
        "ATGGCGTACGACTACGACTGCGACTACGACGATTGA"   # 第三个变化
    ]
    
    # 合并所有序列进行预测
    all_seqs = [original_seq] + variant_seqs
    all_predictions = predictor.predict(all_seqs)
    
    # 输出结果
    print("\n序列比较结果:")
    print(f"原始序列: {original_seq[:20]}...")
    print(f"原始序列翻译效率: {all_predictions[0]:.4f}\n")
    
    for i, (seq, pred) in enumerate(zip(variant_seqs, all_predictions[1:])):
        print(f"变异序列 {i+1}: {seq[:20]}...")
        print(f"翻译效率: {pred:.4f}")
        print(f"相对原始序列变化: {(pred - all_predictions[0]) / all_predictions[0] * 100:.2f}%\n")
    
    return all_predictions

if __name__ == "__main__":
    print("===== RNA翻译效率预测示例 =====")
    
    print("\n1. 单序列预测示例")
    example_single_sequence()
    
    print("\n2. 多序列预测示例")
    example_multiple_sequences()
    
    print("\n3. 序列比较示例")
    example_compare_sequences() 