"""
从E.coli_proteins.csv中提取Value等于2的行
"""
import pandas as pd
import os

# 定义输入和输出文件路径
input_file = "./E.coli_proteins.csv"
output_file = "./E.coli_proteins_value_2.csv"

# 确保输出目录存在
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 读取CSV文件
print(f"正在读取文件: {input_file}")
df = pd.read_csv(input_file)

# 显示原始数据基本信息
print(f"原始数据形状: {df.shape}")
print(f"原始数据列: {df.columns.tolist()}")
print(f"Value列的唯一值: {df['Value'].unique()}")

# 过滤Value等于2的行
filtered_df = df[df['Value'] == 2]
print(f"过滤后数据形状: {filtered_df.shape}")

# 保存结果到新的CSV文件
filtered_df.to_csv(output_file, index=False)
print(f"已将Value=2的行保存到: {output_file}") 