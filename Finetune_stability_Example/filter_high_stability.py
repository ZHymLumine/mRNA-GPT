import argparse
import os
import pandas as pd
import numpy as np
import torch
from rna2sta import RNAStabilityPredictor

def main():
    parser = argparse.ArgumentParser(description="筛选真实稳定性值和预测稳定性值都高于0.5的mRNA序列")
    parser.add_argument("--csv", default="mRNA_Stability.csv", help="输入CSV文件路径，需包含列: Sequence, Value")
    parser.add_argument("--model", default="best_transformer_model.pth", help="训练好的模型路径")
    parser.add_argument("--threshold", type=float, default=0.5, help="稳定性阈值，默认0.5")
    parser.add_argument("--batch_size", type=int, default=16, help="预测批量大小，默认16（避免内存溢出）")
    parser.add_argument("--out", default="high_stability_sequences.csv", help="输出CSV文件路径")
    parser.add_argument("--pred_csv", default="pred_vs_actual_stability_transformer.csv", help="预测结果CSV文件路径")
    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"输入CSV文件不存在: {args.csv}")
    
    print(f"📊 正在读取数据文件: {args.csv}")
    df_original = pd.read_csv(args.csv)
    
    if "Sequence" not in df_original.columns or "Value" not in df_original.columns:
        raise ValueError("输入CSV必须包含列: Sequence 和 Value")
    
    print(f"📈 数据集包含 {len(df_original)} 个序列")
    print(f"📏 稳定性值范围: {df_original['Value'].min():.3f} 到 {df_original['Value'].max():.3f}")
    print(f"📊 平均稳定性值: {df_original['Value'].mean():.3f}")
    
    # 方法1: 如果预测结果文件存在，直接使用
    if os.path.exists(args.pred_csv):
        print(f"🔍 找到预测结果文件: {args.pred_csv}")
        df_pred = pd.read_csv(args.pred_csv)
        
        # 合并原始数据和预测结果
        if len(df_pred) != len(df_original):
            print(f"⚠️  警告: 预测结果数量 ({len(df_pred)}) 与原始数据数量 ({len(df_original)}) 不匹配")
            print("将重新进行预测...")
            use_existing_predictions = False
        else:
            df_combined = df_original.copy()
            df_combined["Predicted"] = df_pred["Predicted"].values
            use_existing_predictions = True
    else:
        use_existing_predictions = False
    
    # 方法2: 如果没有预测结果文件或数量不匹配，重新预测
    if not use_existing_predictions:
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"模型文件不存在: {args.model}")
        
        print(f"🤖 正在加载模型: {args.model}")
        predictor = RNAStabilityPredictor(model_path=args.model)
        
        # 检查是否需要重新处理数据以获取预处理参数
        try:
            # 尝试加载模型，如果失败则需要重新预处理数据
            checkpoint = torch.load(args.model, map_location='cpu')
            if not isinstance(checkpoint, dict) or 'scaler' not in checkpoint:
                print("⚠️  检测到旧格式模型文件，需要预处理少量数据来获取编码器参数...")
                # 只预处理前1000行数据来获取encoder和scaler（节省时间）
                df_sample = df_original.head(1000)
                temp_X, temp_y, temp_splits = predictor.preprocess_data_from_df(df_sample)
                print("✅ 预处理参数已从样本数据生成")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("📝 建议重新训练模型或检查模型文件路径")
            return
        
        print(f"🔮 正在进行预测（批量大小: {args.batch_size}）...")
        print(f"💾 GPU内存使用情况（预测前）: {torch.cuda.memory_allocated()/1024**3:.2f} GB" if torch.cuda.is_available() else "使用CPU进行预测")
        
        sequences = df_original["Sequence"].astype(str).tolist()
        predictions = predictor.predict(sequences, batch_size=args.batch_size)
        
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"💾 GPU内存使用情况（预测后）: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        df_combined = df_original.copy()
        df_combined["Predicted"] = predictions
        
        # 保存预测结果
        pred_result = pd.DataFrame({
            "Actual": df_combined["Value"],
            "Predicted": predictions
        })
        pred_result.to_csv("predictions_for_filtering.csv", index=False)
        print(f"💾 预测结果已保存到: predictions_for_filtering.csv")
    
    # 筛选高稳定性序列
    print(f"🎯 筛选阈值: {args.threshold}")
    
    # 统计信息
    high_actual = df_combined["Value"] > args.threshold
    high_predicted = df_combined["Predicted"] > args.threshold
    both_high = high_actual & high_predicted
    
    print(f"📊 统计信息:")
    print(f"   真实值 > {args.threshold}: {high_actual.sum()} 个序列 ({high_actual.mean()*100:.1f}%)")
    print(f"   预测值 > {args.threshold}: {high_predicted.sum()} 个序列 ({high_predicted.mean()*100:.1f}%)")
    print(f"   两者都 > {args.threshold}: {both_high.sum()} 个序列 ({both_high.mean()*100:.1f}%)")
    
    # 筛选结果
    df_filtered = df_combined[both_high].copy()
    
    if len(df_filtered) == 0:
        print(f"⚠️  没有找到同时满足条件的序列（真实值和预测值都 > {args.threshold}）")
        print("🔧 建议降低阈值或检查模型性能")
        return
    
    # 添加额外信息
    df_filtered["Stability_Difference"] = df_filtered["Predicted"] - df_filtered["Value"]
    df_filtered["Average_Stability"] = (df_filtered["Predicted"] + df_filtered["Value"]) / 2
    
    # 按平均稳定性排序
    df_filtered = df_filtered.sort_values("Average_Stability", ascending=False)
    
    # 保存结果
    df_filtered.to_csv(args.out, index=False)
    
    print(f"✅ 筛选完成！")
    print(f"📁 输出文件: {args.out}")
    print(f"📊 筛选出 {len(df_filtered)} 个高稳定性序列")
    print(f"📈 筛选序列的稳定性范围:")
    print(f"   真实值: {df_filtered['Value'].min():.3f} - {df_filtered['Value'].max():.3f}")
    print(f"   预测值: {df_filtered['Predicted'].min():.3f} - {df_filtered['Predicted'].max():.3f}")
    print(f"   平均稳定性: {df_filtered['Average_Stability'].min():.3f} - {df_filtered['Average_Stability'].max():.3f}")
    
    # 显示前几个序列的信息
    if len(df_filtered) > 0:
        print(f"\n🔝 稳定性最高的5个序列:")
        top_sequences = df_filtered.head(5)
        for i, (idx, row) in enumerate(top_sequences.iterrows(), 1):
            seq_preview = row['Sequence'][:50] + "..." if len(row['Sequence']) > 50 else row['Sequence']
            print(f"   {i}. 真实值: {row['Value']:.3f}, 预测值: {row['Predicted']:.3f}, 平均: {row['Average_Stability']:.3f}")
            print(f"      序列: {seq_preview}")

if __name__ == "__main__":
    main()