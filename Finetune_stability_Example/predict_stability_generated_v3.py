#!/usr/bin/env python3
"""
使用训练好的稳定性预测模型来预测生成序列的稳定性值
支持两个FASTA文件的对比分析，生成分布图和统计分析
"""

import os
import sys
import argparse
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch

# 导入稳定性预测器
sys.path.append('/home/acd13855wx/projects/vita/rna2stab')
from rna2sta import RNAStabilityPredictor

def set_plot_style():
    """设置绘图样式"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 11

def read_fasta(fasta_path: str) -> Tuple[List[str], List[str]]:
    """读取FASTA文件"""
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA文件不存在: {fasta_path}")
    
    sequences = []
    headers = []
    
    with open(fasta_path, 'r') as f:
        current_seq = []
        current_header = None
        
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                # 保存前一个序列
                if current_header is not None and current_seq:
                    sequences.append(''.join(current_seq))
                    headers.append(current_header)
                
                # 开始新序列
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        
        # 保存最后一个序列
        if current_header is not None and current_seq:
            sequences.append(''.join(current_seq))
            headers.append(current_header)
    
    print(f"从 {fasta_path} 读取了 {len(sequences)} 条序列")
    return headers, sequences

def predict_stability_batch(predictor: RNAStabilityPredictor, sequences: List[str], batch_size: int = 16) -> np.ndarray:
    """批量预测序列稳定性"""
    print(f"正在预测 {len(sequences)} 条序列的稳定性值...")
    
    # 将DNA序列转换为RNA（T->U）
    rna_sequences = [seq.replace('T', 'U').upper() for seq in sequences]
    
    # 批量预测
    predictions = predictor.predict(rna_sequences, batch_size=batch_size)
    
    print(f"预测完成！")
    return predictions

def calculate_statistics(values: np.ndarray, label: str) -> Dict:
    """计算统计信息"""
    stats_dict = {
        'label': label,
        'count': len(values),
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values),
        'q25': np.percentile(values, 25),
        'q75': np.percentile(values, 75)
    }
    return stats_dict

def plot_distributions(finetuned_stabilities: np.ndarray, pretrained_stabilities: np.ndarray, 
                      output_dir: str, finetuned_label: str = "Finetuned", 
                      pretrained_label: str = "Pretrained"):
    """绘制分布对比图"""
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 重叠直方图
    axes[0, 0].hist(finetuned_stabilities, bins=50, alpha=0.7, label=finetuned_label, 
                    color='skyblue', density=True)
    axes[0, 0].hist(pretrained_stabilities, bins=50, alpha=0.7, label=pretrained_label, 
                    color='lightcoral', density=True)
    axes[0, 0].set_xlabel('Stability Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Stability Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 并排直方图
    bins = np.linspace(min(np.min(finetuned_stabilities), np.min(pretrained_stabilities)),
                      max(np.max(finetuned_stabilities), np.max(pretrained_stabilities)), 40)
    
    axes[0, 1].hist([finetuned_stabilities, pretrained_stabilities], bins=bins, 
                    label=[finetuned_label, pretrained_label], 
                    color=['skyblue', 'lightcoral'], alpha=0.7)
    axes[0, 1].set_xlabel('Stability Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Stability Distribution Side-by-Side')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 箱线图
    data_for_box = [finetuned_stabilities, pretrained_stabilities]
    box_plot = axes[0, 2].boxplot(data_for_box, labels=[finetuned_label, pretrained_label], 
                                  patch_artist=True)
    box_plot['boxes'][0].set_facecolor('skyblue')
    box_plot['boxes'][1].set_facecolor('lightcoral')
    axes[0, 2].set_ylabel('Stability Value')
    axes[0, 2].set_title('Stability Distribution Box Plot')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 密度曲线
    axes[1, 0].hist(finetuned_stabilities, bins=50, alpha=0.3, density=True, color='skyblue')
    axes[1, 0].hist(pretrained_stabilities, bins=50, alpha=0.3, density=True, color='lightcoral')
    
    # 添加核密度估计
    from scipy.stats import gaussian_kde
    x_range = np.linspace(min(np.min(finetuned_stabilities), np.min(pretrained_stabilities)),
                         max(np.max(finetuned_stabilities), np.max(pretrained_stabilities)), 200)
    
    kde_finetuned = gaussian_kde(finetuned_stabilities)
    kde_pretrained = gaussian_kde(pretrained_stabilities)
    
    axes[1, 0].plot(x_range, kde_finetuned(x_range), color='blue', linewidth=2, label=f'{finetuned_label} KDE')
    axes[1, 0].plot(x_range, kde_pretrained(x_range), color='red', linewidth=2, label=f'{pretrained_label} KDE')
    axes[1, 0].set_xlabel('Stability Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Kernel Density Estimation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Q-Q图
    from scipy import stats
    stats.probplot(finetuned_stabilities, dist="norm", plot=axes[1, 1])
    axes[1, 1].get_lines()[0].set_markerfacecolor('skyblue')
    axes[1, 1].get_lines()[0].set_markeredgecolor('blue')
    axes[1, 1].get_lines()[0].set_markersize(4)
    axes[1, 1].set_title(f'{finetuned_label} Q-Q Plot (Normal)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 累积分布函数
    sorted_finetuned = np.sort(finetuned_stabilities)
    sorted_pretrained = np.sort(pretrained_stabilities)
    
    y_finetuned = np.arange(1, len(sorted_finetuned) + 1) / len(sorted_finetuned)
    y_pretrained = np.arange(1, len(sorted_pretrained) + 1) / len(sorted_pretrained)
    
    axes[1, 2].plot(sorted_finetuned, y_finetuned, color='blue', linewidth=2, label=finetuned_label)
    axes[1, 2].plot(sorted_pretrained, y_pretrained, color='red', linewidth=2, label=pretrained_label)
    axes[1, 2].set_xlabel('Stability Value')
    axes[1, 2].set_ylabel('Cumulative Probability')
    axes[1, 2].set_title('Cumulative Distribution Function')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stability_distribution_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_violin_comparison(finetuned_stabilities: np.ndarray, pretrained_stabilities: np.ndarray, 
                          output_dir: str, finetuned_label: str = "Finetuned", 
                          pretrained_label: str = "Pretrained"):
    """绘制小提琴图对比"""
    
    # 准备数据
    data = pd.DataFrame({
        'Stability': np.concatenate([finetuned_stabilities, pretrained_stabilities]),
        'Model': [finetuned_label] * len(finetuned_stabilities) + [pretrained_label] * len(pretrained_stabilities)
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 小提琴图
    sns.violinplot(data=data, x='Model', y='Stability', ax=ax1, palette=['skyblue', 'lightcoral'])
    ax1.set_title('Stability Distribution Violin Plot')
    ax1.grid(True, alpha=0.3)
    
    # 条形图：平均值对比
    means = [np.mean(finetuned_stabilities), np.mean(pretrained_stabilities)]
    stds = [np.std(finetuned_stabilities), np.std(pretrained_stabilities)]
    
    bars = ax2.bar([finetuned_label, pretrained_label], means, 
                   yerr=stds, capsize=5, alpha=0.7, color=['skyblue', 'lightcoral'])
    ax2.set_ylabel('Mean Stability Value')
    ax2.set_title('Mean Stability Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, mean, std in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stability_violin_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def perform_statistical_tests(finetuned_stabilities: np.ndarray, pretrained_stabilities: np.ndarray) -> Dict:
    """执行统计检验"""
    
    # T检验
    t_stat, t_pvalue = stats.ttest_ind(finetuned_stabilities, pretrained_stabilities)
    
    # Mann-Whitney U检验（非参数）
    u_stat, u_pvalue = stats.mannwhitneyu(finetuned_stabilities, pretrained_stabilities, 
                                         alternative='two-sided')
    
    # Kolmogorov-Smirnov检验
    ks_stat, ks_pvalue = stats.ks_2samp(finetuned_stabilities, pretrained_stabilities)
    
    # 效应量 (Cohen's d)
    pooled_std = np.sqrt((np.var(finetuned_stabilities) + np.var(pretrained_stabilities)) / 2)
    cohens_d = (np.mean(finetuned_stabilities) - np.mean(pretrained_stabilities)) / pooled_std
    
    return {
        't_statistic': t_stat,
        't_pvalue': t_pvalue,
        'mannwhitney_u': u_stat,
        'mannwhitney_pvalue': u_pvalue,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'cohens_d': cohens_d
    }

def save_results(finetuned_headers: List[str], finetuned_sequences: List[str], finetuned_stabilities: np.ndarray,
                pretrained_headers: List[str], pretrained_sequences: List[str], pretrained_stabilities: np.ndarray,
                output_dir: str, finetuned_label: str = "Finetuned", pretrained_label: str = "Pretrained"):
    """保存预测结果和统计分析"""
    
    # 保存详细结果
    finetuned_df = pd.DataFrame({
        'Header': finetuned_headers,
        'Sequence': finetuned_sequences,
        'Stability': finetuned_stabilities,
        'Sequence_Length': [len(seq) for seq in finetuned_sequences],
        'Model_Type': finetuned_label
    })
    
    pretrained_df = pd.DataFrame({
        'Header': pretrained_headers,
        'Sequence': pretrained_sequences,
        'Stability': pretrained_stabilities,
        'Sequence_Length': [len(seq) for seq in pretrained_sequences],
        'Model_Type': pretrained_label
    })
    
    # 合并数据
    combined_df = pd.concat([finetuned_df, pretrained_df], ignore_index=True)
    combined_df.to_csv(os.path.join(output_dir, 'stability_predictions.csv'), index=False)
    
    # 计算统计信息
    finetuned_stats = calculate_statistics(finetuned_stabilities, finetuned_label)
    pretrained_stats = calculate_statistics(pretrained_stabilities, pretrained_label)
    
    # 执行统计检验
    statistical_tests = perform_statistical_tests(finetuned_stabilities, pretrained_stabilities)
    
    # 保存统计报告
    with open(os.path.join(output_dir, 'stability_analysis_report.txt'), 'w') as f:
        f.write("mRNA序列稳定性预测分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 数据概览\n")
        f.write("-" * 20 + "\n")
        f.write(f"{finetuned_label}序列数量: {finetuned_stats['count']}\n")
        f.write(f"{pretrained_label}序列数量: {pretrained_stats['count']}\n\n")
        
        f.write("2. 描述性统计\n")
        f.write("-" * 20 + "\n")
        f.write(f"{finetuned_label}:\n")
        f.write(f"  平均值: {finetuned_stats['mean']:.4f}\n")
        f.write(f"  标准差: {finetuned_stats['std']:.4f}\n")
        f.write(f"  中位数: {finetuned_stats['median']:.4f}\n")
        f.write(f"  范围: {finetuned_stats['min']:.4f} - {finetuned_stats['max']:.4f}\n")
        f.write(f"  四分位数: {finetuned_stats['q25']:.4f} - {finetuned_stats['q75']:.4f}\n\n")
        
        f.write(f"{pretrained_label}:\n")
        f.write(f"  平均值: {pretrained_stats['mean']:.4f}\n")
        f.write(f"  标准差: {pretrained_stats['std']:.4f}\n")
        f.write(f"  中位数: {pretrained_stats['median']:.4f}\n")
        f.write(f"  范围: {pretrained_stats['min']:.4f} - {pretrained_stats['max']:.4f}\n")
        f.write(f"  四分位数: {pretrained_stats['q25']:.4f} - {pretrained_stats['q75']:.4f}\n\n")
        
        f.write("3. 统计检验\n")
        f.write("-" * 20 + "\n")
        f.write(f"独立样本t检验:\n")
        f.write(f"  t统计量: {statistical_tests['t_statistic']:.4f}\n")
        f.write(f"  p值: {statistical_tests['t_pvalue']:.6f}\n\n")
        
        f.write(f"Mann-Whitney U检验:\n")
        f.write(f"  U统计量: {statistical_tests['mannwhitney_u']:.4f}\n")
        f.write(f"  p值: {statistical_tests['mannwhitney_pvalue']:.6f}\n\n")
        
        f.write(f"Kolmogorov-Smirnov检验:\n")
        f.write(f"  KS统计量: {statistical_tests['ks_statistic']:.4f}\n")
        f.write(f"  p值: {statistical_tests['ks_pvalue']:.6f}\n\n")
        
        f.write(f"效应量 (Cohen's d): {statistical_tests['cohens_d']:.4f}\n")
        
        f.write("\n4. 结果解释\n")
        f.write("-" * 20 + "\n")
        if statistical_tests['t_pvalue'] < 0.05:
            f.write("两组间存在显著差异 (p < 0.05)\n")
        else:
            f.write("两组间无显著差异 (p >= 0.05)\n")
        
        if abs(statistical_tests['cohens_d']) < 0.2:
            effect_size = "小"
        elif abs(statistical_tests['cohens_d']) < 0.8:
            effect_size = "中等"
        else:
            effect_size = "大"
        f.write(f"效应量大小: {effect_size}\n")

def main():
    parser = argparse.ArgumentParser(description="预测生成序列的稳定性值并进行对比分析")
    parser.add_argument("--model", default="/home/acd13855wx/projects/vita/rna2stab/best_transformer_model.pth", 
                       help="稳定性预测模型路径")
    parser.add_argument("--finetuned_fasta", 
                       default="/home/acd13855wx/projects/vita/rna2stab/finetune/finetune_generated/finetuned_generated.fasta", 
                       help="微调生成的序列FASTA文件")
    parser.add_argument("--pretrained_fasta", 
                       default="/home/acd13855wx/projects/vita/rna2stab/finetune/pretrained_generated/pretrained_generated.fasta", 
                       help="预训练生成的序列FASTA文件")
    parser.add_argument("--output", default="/home/acd13855wx/projects/vita/rna2stab/stability_comparison", 
                       help="输出目录")
    parser.add_argument("--batch_size", type=int, default=16, help="预测批量大小")
    parser.add_argument("--finetuned_label", default="Finetuned", help="微调模型标签")
    parser.add_argument("--pretrained_label", default="Pretrained", help="预训练模型标签")
    parser.add_argument("--original_csv", default="/home/acd13855wx/projects/vita/rna2stab/mRNA_Stability.csv", 
                       help="原始训练数据集路径（用于获取预处理参数）")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    print(f"📁 输出目录: {args.output}")
    
    # 设置绘图样式
    set_plot_style()
    
    # 加载稳定性预测模型
    print(f"🤖 加载稳定性预测模型: {args.model}")
    predictor = RNAStabilityPredictor(model_path=args.model)
    
    # 检查是否需要预处理参数，如果需要则从原始数据集获取
    try:
        # 尝试加载模型检查格式
        checkpoint = torch.load(args.model, map_location='cpu')
        if not isinstance(checkpoint, dict) or 'scaler' not in checkpoint:
            print("⚠️  检测到旧格式模型文件，需要从原始数据集获取预处理参数...")
            # 从原始数据集获取预处理参数
            if os.path.exists(args.original_csv):
                print(f"📖 从原始数据集获取预处理参数: {args.original_csv}")
                import pandas as pd
                df_sample = pd.read_csv(args.original_csv).head(1000)  # 只用前1000行来获取参数
                temp_X, temp_y, temp_splits = predictor.preprocess_data_from_df(df_sample)
                print("✅ 预处理参数已生成")
            else:
                raise FileNotFoundError(f"原始数据集文件不存在: {args.original_csv}")
    except Exception as e:
        print(f"❌ 模型处理失败: {e}")
        raise
    
    # 读取FASTA文件
    print(f"📖 读取微调生成序列: {args.finetuned_fasta}")
    finetuned_headers, finetuned_sequences = read_fasta(args.finetuned_fasta)
    
    print(f"📖 读取预训练生成序列: {args.pretrained_fasta}")
    pretrained_headers, pretrained_sequences = read_fasta(args.pretrained_fasta)
    
    # 预测稳定性
    print("🔮 预测微调序列稳定性...")
    finetuned_stabilities = predict_stability_batch(predictor, finetuned_sequences, args.batch_size)
    
    print("🔮 预测预训练序列稳定性...")
    pretrained_stabilities = predict_stability_batch(predictor, pretrained_sequences, args.batch_size)
    
    # 生成对比图表
    print("📊 生成分布对比图...")
    plot_distributions(finetuned_stabilities, pretrained_stabilities, args.output, 
                      args.finetuned_label, args.pretrained_label)
    
    print("📊 生成小提琴图对比...")
    plot_violin_comparison(finetuned_stabilities, pretrained_stabilities, args.output, 
                          args.finetuned_label, args.pretrained_label)
    
    # 保存结果和统计分析
    print("💾 保存结果和统计分析...")
    save_results(finetuned_headers, finetuned_sequences, finetuned_stabilities,
                pretrained_headers, pretrained_sequences, pretrained_stabilities,
                args.output, args.finetuned_label, args.pretrained_label)
    
    # 输出简要统计
    print("\n📈 简要统计:")
    print(f"{args.finetuned_label}: 平均={np.mean(finetuned_stabilities):.4f}, 标准差={np.std(finetuned_stabilities):.4f}")
    print(f"{args.pretrained_label}: 平均={np.mean(pretrained_stabilities):.4f}, 标准差={np.std(pretrained_stabilities):.4f}")
    
    # 执行t检验
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(finetuned_stabilities, pretrained_stabilities)
    print(f"t检验: t={t_stat:.4f}, p={p_value:.6f}")
    
    print(f"\n✅ 分析完成！结果保存在 {args.output}")
    print("生成的文件:")
    print("  - stability_predictions.csv: 详细预测结果")
    print("  - stability_analysis_report.txt: 统计分析报告")
    print("  - stability_distribution_comparison.png: 分布对比图")
    print("  - stability_violin_comparison.png: 小提琴图对比")

if __name__ == "__main__":
    main()
