import pandas as pd
import os
from collections import Counter

def analyze_cds_start_codons(csv_path):
    """
    分析CSV文件中CDS_sequence列的前3个字符的唯一形式
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        start_codons: 不同的起始密码子及其计数
    """
    print(f"读取文件: {csv_path}")
    
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: 文件不存在: {csv_path}")
        return None
    
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 检查是否包含CDS_sequence列
        if 'CDS_sequence' not in df.columns:
            print(f"错误: CSV文件中没有'CDS_sequence'列")
            print(f"可用的列: {df.columns.tolist()}")
            return None
        
        # 提取每个序列的前3个字符
        start_codons = []
        for seq in df['CDS_sequence']:
            if isinstance(seq, str) and len(seq) >= 3:
                start_codons.append(seq[:3])
            else:
                print(f"警告: 跳过无效序列: {seq}")
        
        # 计算不同起始密码子的数量和频率
        codon_counter = Counter(start_codons)
        
        print(f"\n总序列数: {len(start_codons)}")
        print(f"不同起始密码子数量: {len(codon_counter)}")
        
        print("\n起始密码子统计:")
        print("-" * 30)
        print("起始密码子  |  数量  |  频率(%)")
        print("-" * 30)
        
        for codon, count in codon_counter.most_common():
            percentage = (count / len(start_codons)) * 100
            print(f"{codon:^12}|{count:^8}|{percentage:^10.2f}")
        
        return codon_counter
    
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return None

def main():
    csv_path = "/home/yzhang/research/mRNAdesigner_3/predictive_model/mRNA2te/ecoli_TE_CDS_final.csv"
    
    # 分析起始密码子
    codon_counter = analyze_cds_start_codons(csv_path)
    
    if codon_counter:
        # 分析是否所有序列都以ATG开始
        if len(codon_counter) == 1 and 'ATG' in codon_counter:
            print("\n所有序列都以ATG开始")
        else:
            print("\n非ATG起始密码子:")
            for codon, count in codon_counter.items():
                if codon != 'ATG':
                    print(f"  {codon}: {count}个序列")
    
        # 保存结果到文件
        output_file = "cds_start_codon_analysis.txt"
        with open(output_file, 'w') as f:
            f.write("CDS序列起始密码子分析\n")
            f.write("-" * 30 + "\n")
            f.write(f"总序列数: {sum(codon_counter.values())}\n")
            f.write(f"不同起始密码子数量: {len(codon_counter)}\n\n")
            
            f.write("起始密码子统计:\n")
            f.write("-" * 30 + "\n")
            f.write("起始密码子  |  数量  |  频率(%)\n")
            f.write("-" * 30 + "\n")
            
            total = sum(codon_counter.values())
            for codon, count in codon_counter.most_common():
                percentage = (count / total) * 100
                f.write(f"{codon:^12}|{count:^8}|{percentage:^10.2f}\n")
        
        print(f"\n分析结果已保存到: {output_file}")

if __name__ == "__main__":
    main() 