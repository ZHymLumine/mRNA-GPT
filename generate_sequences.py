import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizerFast
import pickle
import json
from datetime import datetime

from model import GPT, GPTConfig
from predictive_model.mRNA2te.rna2te import RNATranslationEfficiencyPredictor



class PPOConfig:
    def __init__(self):
        self.learning_rate = 1e-5
        self.critic_learning_rate = 1e-4
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.ppo_epochs = 5
        self.mini_batch_size = 16
        self.generate_batch_size = 16
        self.max_seq_len = 512
        self.max_new_tokens = 512
        self.top_k = 50
        self.top_p = 0.95
        self.temperature = 0.7
        self.target_te = 1.0  # 目标TE值
        self.entropy_coef = 0.01
        self.max_iterations = 1000
        self.save_interval = 10
        self.te_reward_scale = 10.0  # TE奖励缩放因子
        self.repetition_penalty = 1.0
        self.output_dir = "ppo_output"  # 默认输出目录


def load_te_model():
    """加载翻译效率预测模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = "predictive_model/mRNA2te/best_transformer_model.pth"
    
    predictor = RNATranslationEfficiencyPredictor(model_path=model_path)
    predictor.preprocess_data("predictive_model/mRNA2te/ecoli_TE_CDS_final.csv")
    predictor.device = device
    if predictor.model is None:
        predictor.model = predictor.build_model()
        predictor.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        predictor.model.eval()
    
    return predictor


def generate_sequences(model, tokenizer, num_sequences, config, device):
    """生成指定数量的mRNA序列
    
    Args:
        model: GPT模型
        tokenizer: 分词器
        num_sequences: 需要生成的序列数量
        config: 生成配置
        device: 设备
        
    Returns:
        generated_sequences: 生成的token序列列表
        decoded_sequences: 解码后的文本序列列表
    """
    model.eval()
    generated_sequences = []
    decoded_sequences = []
    
    # 定义有效的起始密码子及其频率
    valid_start_codons = {
        "ATG": 90.59,
        "GTG": 7.42,
        "TTG": 1.88,
        "CTG": 0.05,
        "ATT": 0.05
    }
    
    # 按照频率创建起始密码子选择器
    start_codon_choices = []
    start_codon_weights = []
    for codon, freq in valid_start_codons.items():
        start_codon_choices.append(codon)
        start_codon_weights.append(freq)
    
    # 归一化权重
    total_weight = sum(start_codon_weights)
    start_codon_weights = [w/total_weight for w in start_codon_weights]
    
    batch_size = config.batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, num_sequences, batch_size), desc="生成序列"):
            current_batch_size = min(batch_size, num_sequences - len(generated_sequences))
            if current_batch_size <= 0:
                break
            
            # 使用[SEP]作为起始token
            start = f"[SEP]"
            start_ids = tokenizer.encode("".join(start), add_special_tokens=False)
            
            # 创建批量输入 - 为每个样本复制相同的起始token
            batch_input = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0).repeat(current_batch_size, 1)
            
            # 批量生成序列
            output_ids = model.generate(
                batch_input, 
                config.max_new_tokens, 
                strategy='top_k', 
                temperature=config.temperature, 
                top_k=config.top_k, 
                repetition_penalty=config.repetition_penalty
            )
            
            # 处理生成的每个序列
            for j in range(current_batch_size):
                seq_ids = output_ids[j].tolist()
                
                seq_text = tokenizer.decode(seq_ids, skip_special_tokens=True).replace(' ', '').translate(str.maketrans('U', 'T'))
                
                if len(seq_text) >= 3:
                    start_codon = seq_text[:3]
                    if start_codon not in valid_start_codons:
                        # 按照频率分布随机选择一个有效的起始密码子
                        new_start_codon = np.random.choice(start_codon_choices, p=start_codon_weights)
                        seq_text = new_start_codon + seq_text[3:]
                        print(f"序列 {len(generated_sequences) + j + 1} 的起始密码子 '{start_codon}' 无效，已替换为 '{new_start_codon}'")
                else:
                    new_start_codon = np.random.choice(start_codon_choices, p=start_codon_weights)
                    seq_text = new_start_codon + seq_text
                    print(f"序列 {len(generated_sequences) + j + 1} 太短，已添加起始密码子 '{new_start_codon}'")
                
                if (len(generated_sequences) + j) % 10 == 0:
                    print(f"序列 {len(generated_sequences) + j + 1} 示例: {seq_text[:50]}...")
                
                generated_sequences.append(output_ids[j])
                decoded_sequences.append(seq_text)
            
            print(f"已生成 {len(decoded_sequences)}/{num_sequences} 个序列")
    
    return generated_sequences, decoded_sequences


def evaluate_sequences(sequences, te_model):
    """评估序列的翻译效率
    
    Args:
        sequences: 序列列表
        te_model: 翻译效率预测模型
        
    Returns:
        te_values: 预测的TE值列表
    """
    print("正在评估序列翻译效率...")
    te_values = te_model.predict(sequences)
    
    return te_values


def save_sequences(sequences, te_values, output_dir):
    """保存生成的序列和对应的TE值
    
    Args:
        sequences: 序列列表
        te_values: TE值列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建包含序列和TE值的列表
    sequence_data = []
    for seq, te in zip(sequences, te_values):
        sequence_data.append({
            'sequence': seq,
            'te_score': float(te)
        })
    
    # 按TE值排序
    sequence_data.sort(key=lambda x: x['te_score'], reverse=True)
    
    
    # 保存为CSV格式（所有序列，更容易导入到其他工具中）
    csv_path = os.path.join(output_dir, "generated_sequences.csv")
    with open(csv_path, 'w') as f:
        f.write("序号,TE值,序列\n")
        for i, item in enumerate(sequence_data):
            f.write(f"{i+1},{item['te_score']:.6f},{item['sequence']}\n")
    
    # 保存为文本格式，包含所有序列
    txt_path = os.path.join(output_dir, "generated_sequences.txt")
    with open(txt_path, 'w') as f:
        f.write(f"总共生成了 {len(sequence_data)} 个序列\n\n")
        f.write(f"TE值统计信息:\n")
        f.write(f"  平均值: {np.mean(te_values):.4f}\n")
        f.write(f"  最大值: {np.max(te_values):.4f}\n")
        f.write(f"  最小值: {np.min(te_values):.4f}\n")
        f.write(f"  中位数: {np.median(te_values):.4f}\n\n")
        
        f.write(f"所有序列 (按TE值降序排列):\n\n")
        for i, item in enumerate(sequence_data):
            f.write(f"序列 {i+1}, TE值: {item['te_score']:.4f}\n")
            f.write(f"{item['sequence']}\n\n")
    
    # 保存统计信息到JSON
    stats_path = os.path.join(output_dir, "te_statistics.json")
    te_stats = {
        'mean': float(np.mean(te_values)),
        'max': float(np.max(te_values)),
        'min': float(np.min(te_values)),
        'std': float(np.std(te_values)),
        'median': float(np.median(te_values)),
        'percentiles': {
            '10': float(np.percentile(te_values, 10)),
            '25': float(np.percentile(te_values, 25)),
            '50': float(np.percentile(te_values, 50)),
            '75': float(np.percentile(te_values, 75)),
            '90': float(np.percentile(te_values, 90)),
        }
    }
    with open(stats_path, 'w') as f:
        json.dump(te_stats, f, indent=2)
    
    print(f"所有序列已保存到以下文件:")
    print(f"  - CSV格式: {csv_path}")
    print(f"  - 摘要文本: {txt_path}")
    print(f"TE统计数据已保存到: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="使用PPO微调后的模型生成mRNA序列")
    parser.add_argument("--model_path", type=str, required=True, help="微调模型路径")
    parser.add_argument("--num_sequences", type=int, default=1000, help="要生成的序列数量")
    parser.add_argument("--output_dir", type=str, default="generated_sequences", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=16, help="生成批次大小")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top_k", type=int, default=50, help="top-k采样参数")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="重复惩罚系数")
    parser.add_argument("--debug", action="store_true", help="启用调试模式，打印更多信息")
    parser.add_argument('--checkpoint', type=str, default="output/ckpt_563000.pt", help='预训练模型检查点路径')
    parser.add_argument('--ppo', action="store_true", help='是否使用PPO微调模型')
    parser.add_argument('--sft', action="store_true", help='是否使用SFT微调模型')
    args = parser.parse_args()
    
    # 创建运行ID和输出目录
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{run_id}")
    if args.ppo:
        output_dir = output_dir + "_ppo"
    elif args.sft:
        output_dir = output_dir + "_sft"
    else:
        output_dir = output_dir + "_pretrained"
    os.makedirs(output_dir, exist_ok=True)
    
    # save generation config
    with open(os.path.join(output_dir, "generation_config.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = BertTokenizerFast(vocab_file="tokenizer/vocab.txt", do_lower_case=False)
    
    try:
        print(f"加载模型: {args.model_path}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        print(checkpoint)
        gptconf = GPTConfig(**checkpoint['model_args'])
        actor_model = GPT(gptconf).to(device)


        if args.ppo:
            print(f"使用PPO微调模型加载权重")
            ppo_checkpoint = torch.load(args.model_path, map_location=device)
            if 'actor_state_dict' in ppo_checkpoint:
                print("使用actor_state_dict加载权重")

                actor_model.load_state_dict(ppo_checkpoint['actor_state_dict'])
        elif args.sft:
            print(f"使用SFT微调模型加载权重")
            sft_checkpoint = torch.load(args.model_path, map_location=device)
            state_dict = sft_checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
            # 使用strict=False忽略额外的te_head参数
            actor_model.load_state_dict(state_dict, strict=False)
        else:
            print(f"使用原始预训练模型加载权重")
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            actor_model.load_state_dict(state_dict)

        class GenConfig:
            def __init__(self):
                self.batch_size = args.batch_size
                self.temperature = args.temperature
                self.top_k = args.top_k
                self.max_new_tokens = args.max_length
                self.repetition_penalty = args.repetition_penalty
        
        gen_config = GenConfig()
        
        # 加载TE评估模型
        te_model = load_te_model()
        
        # 生成序列
        print(f"开始生成 {args.num_sequences} 个序列...")
        _, decoded_sequences = generate_sequences(actor_model, tokenizer, args.num_sequences, gen_config, device)
        
        # 评估序列的翻译效率
        te_values = evaluate_sequences(decoded_sequences, te_model)
        
        # 保存序列和评估结果
        save_sequences(decoded_sequences, te_values, output_dir)
        
        # 输出最高TE序列
        max_te_idx = np.argmax(te_values)
        print(f"\n最高TE序列 (TE = {te_values[max_te_idx]:.4f}):")
        print(decoded_sequences[max_te_idx][:100] + ("..." if len(decoded_sequences[max_te_idx]) > 100 else ""))
    
    except Exception as e:
        error_log_path = os.path.join(output_dir, "error_log.txt")
        with open(error_log_path, 'w') as f:
            f.write(f"加载或生成过程中出错: {str(e)}\n")
            import traceback
            traceback.print_exc(file=f)
        print(f"错误详情已保存到 {error_log_path}")
        raise


if __name__ == "__main__":
    main() 

# 使用示例
# CUDA_VISIBLE_DEVICES=0 python generate_sequences.py --model_path ppo_output/run_<时间戳>/models/final_model.pt --num_sequences 1000 --output_dir ppo_generated_results
# 调试模式
# CUDA_VISIBLE_DEVICES=0 python generate_sequences.py --model_path ppo_output/run_<时间戳>/models/final_model.pt --num_sequences 10 --output_dir ppo_generated_results -

# CUDA_VISIBLE_DEVICES=5 python generate_sequences.py --model_path /home/yzhang/research/mRNAdesigner_3/output/sft_te/sft_ckpt_best.pt --num_sequences 1000 --output_dir sft_generated_results --sft