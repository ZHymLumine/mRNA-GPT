import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.distributions import Categorical
from transformers import BertTokenizerFast
import pandas as pd
import pickle
import random
from collections import deque
from datetime import datetime

from model import GPT, GPTConfig
from predictive_model.mRNA2te.rna2te import TransformerRegressor, RNATranslationEfficiencyPredictor


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

# PPO的Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self, actor_model, critic_config):
        super(ActorCritic, self).__init__()
        self.actor = actor_model
        
        # 值网络（Critic）
        self.critic = nn.Sequential(
            nn.Linear(critic_config.n_embd, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, tokens, targets=None):
        # actor输出动作概率分布
        logits, _ = self.actor(tokens, targets)
        return logits
    
    def get_value(self, tokens):
        # 用actor的编码层特征作为critic的输入
        with torch.no_grad():
            # 获取最后一层Transformer的输出
            x = self.actor.transformer.wte(tokens)
            pos = torch.arange(0, tokens.size(1), dtype=torch.long, device=tokens.device)
            pos_emb = self.actor.transformer.wpe(pos)
            x = self.actor.transformer.drop(x + pos_emb)
            for block in self.actor.transformer.h:
                x = block(x)
            x = self.actor.transformer.ln_f(x)
            
            # 使用序列的平均特征
            pooled_output = x.mean(dim=1)
            
        # 计算值函数
        value = self.critic(pooled_output)
        return value

class SequenceStorage:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, sequence, decoded_sequence, te_score):
        self.buffer.append((sequence, decoded_sequence, te_score))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class PPOTrainer:
    def __init__(self, config, actor_model, reward_model, tokenizer, device):
        self.config = config
        self.actor_model = actor_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device
        
        # 创建ActorCritic网络
        self.ac_model = ActorCritic(actor_model, actor_model.config).to(device)
        
        # 优化器
        self.optimizer = Adam([
            {'params': self.ac_model.actor.parameters(), 'lr': config.learning_rate},
            {'params': self.ac_model.critic.parameters(), 'lr': config.critic_learning_rate}
        ])
        
        # 存储生成的高TE序列
        self.sequence_storage = SequenceStorage()
        
        # 每轮更新的统计数据
        self.stats = {
            'mean_reward': [],
            'mean_te': [],
            'mean_length': [],
            'max_te': 0.0,
            'best_sequence': None
        }
        
        # 创建输出目录结构
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.output_dir, f"run_{self.run_id}")
        self.model_dir = os.path.join(self.output_dir, "models")
        self.sequence_dir = os.path.join(self.output_dir, "sequences")
        self.log_dir = os.path.join(self.output_dir, "logs")
        
        # 确保目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.sequence_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建日志文件
        self.log_file = os.path.join(self.log_dir, "training_log.txt")
        with open(self.log_file, 'w') as f:
            f.write(f"PPO训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"配置参数:\n")
            for key, value in vars(config).items():
                f.write(f"  {key}: {value}\n")
    
    def generate_sequences(self, num_sequences=16):
        """生成多个mRNA序列"""
        self.actor_model.eval()
        generated_sequences = []
        decoded_sequences = []  # 新增：存储解码后的序列
        
        with torch.no_grad():
            for _ in range((num_sequences + self.config.generate_batch_size - 1) // self.config.generate_batch_size):
                batch_size = min(self.config.generate_batch_size, num_sequences - len(generated_sequences))
                if batch_size <= 0:
                    break
                
                start = f"[SEP]"
                start_ids = self.tokenizer.encode("".join(start), add_special_tokens=False)
                
                # 创建批量输入 - 为每个样本复制相同的起始token
                batch_input = torch.tensor(start_ids, dtype=torch.long, device=self.device).unsqueeze(0).repeat(batch_size, 1)
                
                # 批量生成序列
                output_ids = self.actor_model.generate(
                    batch_input, 
                    self.config.max_new_tokens, 
                    strategy='top_k', 
                    temperature=self.config.temperature, 
                    top_k=self.config.top_k, 
                    repetition_penalty=self.config.repetition_penalty
                )
                
                for i in range(batch_size):
                    seq_ids = output_ids[i].tolist()
                    
                    # 解码序列
                    seq_text = self.tokenizer.decode(seq_ids, skip_special_tokens=True).replace(' ', '').translate(str.maketrans('U', 'T'))
                    
                    # 每8个序列打印一次示例信息
                    if len(generated_sequences) % 8 == 0 and i == 0:
                        print(f"第{len(generated_sequences)}个序列示例 output_ids: {seq_ids[:10]}...")
                        print(f"序列示例: {seq_text[:50]}...")
                    
                    # 添加到序列列表中
                    generated_sequences.append(output_ids[i])
                    decoded_sequences.append(seq_text)
                
                print(f"已生成 {len(generated_sequences)}/{num_sequences} 个序列")
        
        return generated_sequences, decoded_sequences
    
    def calculate_reward(self, decoded_sequences):
        """计算每个序列的TE值和奖励，使用RNATranslationEfficiencyPredictor

        Args:
            decoded_sequences: 已解码的DNA序列列表，每个元素是一个字符串

        Returns:
            rewards: 每个序列的奖励值列表
            te_values: 每个序列的TE预测值列表
        """
        rewards = []
        te_values = []
        
        # 使用predict方法获取翻译效率预测值
        te_predictions = self.reward_model.predict(decoded_sequences)
        
        # 对每个预测值计算奖励
        for te_pred in te_predictions:
            te_values.append(te_pred)
            
            # 计算奖励 - 使用配置中的缩放因子
            reward = te_pred * self.config.te_reward_scale
            rewards.append(reward)
            
            # 更新最佳序列记录
            if te_pred > self.stats['max_te']:
                idx = len(te_values) - 1
                self.stats['max_te'] = te_pred
                self.stats['best_sequence'] = decoded_sequences[idx]
                
            # 保存高TE的序列
            if te_pred > 0.5:  # 可调整阈值
                idx = len(te_values) - 1
                self.sequence_storage.add('', decoded_sequences[idx], te_pred)
        
        return rewards, te_values
    
    def ppo_update(self, sequences, old_log_probs, rewards, advantages):
        """执行PPO更新"""
        self.actor_model.train()
        
        for _ in range(self.config.ppo_epochs):
            # 计算所有序列的新动作概率和值
            for i in range(0, len(sequences), self.config.mini_batch_size):
                batch_sequences = sequences[i:i+self.config.mini_batch_size]
                # 确保所有张量都是float32类型
                batch_old_log_probs = torch.cat(old_log_probs[i:i+self.config.mini_batch_size], dim=0).float()
                batch_rewards = rewards[i:i+self.config.mini_batch_size].float()
                batch_advantages = advantages[i:i+self.config.mini_batch_size].float()
                
                # 计算策略损失
                logits = []
                values = []
                entropy = 0
                for seq in batch_sequences:
                    inputs = seq[:-1].unsqueeze(0)
                    targets = seq[1:].unsqueeze(0)
                    
                    # 获取logits和value
                    seq_logits = self.ac_model(inputs, targets)
                    seq_value = self.ac_model.get_value(inputs)
                    
                    logits.append(seq_logits)
                    values.append(seq_value)
                    
                    # 计算熵
                    probs = F.softmax(seq_logits, dim=-1)
                    entropy += -torch.sum(probs * torch.log(probs + 1e-10)) / probs.size(0)
                
                # 计算新的动作概率
                new_log_probs = torch.cat([l.view(-1, l.size(-1)).log_softmax(-1) for l in logits], dim=0).float()
                values = torch.cat(values).float()
                
                # 计算比率
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 将batch_advantages扩展到与ratios相同的维度
                seq_lengths = [len(seq) - 1 for seq in batch_sequences]
                expanded_advantages = []
                for adv, seq_len in zip(batch_advantages, seq_lengths):
                    expanded_advantages.append(adv.repeat(seq_len))
                expanded_advantages = torch.cat(expanded_advantages).float()
                expanded_advantages = expanded_advantages.view(-1, 1)
                
                # 计算裁剪后的目标
                surr1 = ratios * expanded_advantages
                surr2 = torch.clamp(ratios, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * expanded_advantages
                
                # 计算actor损失：取min以实现裁剪
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 计算critic损失：值函数
                critic_loss = F.mse_loss(values.squeeze(), batch_rewards)
                
                # 计算总损失
                loss = actor_loss + 0.5 * critic_loss - self.config.entropy_coef * entropy
                
                # 梯度更新
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(), 0.5)
                self.optimizer.step()
    
    def compute_advantages(self, rewards, values):
        """计算GAE优势函数"""
        advantages = []
        returns = []
        gae = 0
        
        # 计算每个序列的优势
        for i in range(len(rewards)):
            # 使用GAE计算优势
            delta = rewards[i] - values[i]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages.append(gae)
            returns.append(gae + values[i])
        
        # 标准化优势
        advantages = torch.tensor(advantages, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, torch.tensor(returns, device=self.device)
    
    def train_iteration(self):
        """执行一次完整的PPO训练迭代"""
        # 生成序列
        sequences, decoded_sequences = self.generate_sequences(self.config.generate_batch_size)
        if len(decoded_sequences) == 0:
            return False
        
        # 计算序列的奖励
        rewards, te_values = self.calculate_reward(decoded_sequences)

        if len(rewards) == 0:
            return False
        
        old_log_probs = []
        values = []
        
        with torch.no_grad():
            for seq in sequences:
                inputs = seq[:-1].unsqueeze(0)
                targets = seq[1:].unsqueeze(0)
                
                # 获取logits和value
                logits, _ = self.actor_model(inputs, targets)
                value = self.ac_model.get_value(inputs)
                
                # 计算log概率
                log_probs = F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1)
                old_log_probs.append(log_probs)
                values.append(value.item())
        
        # 计算优势函数
        advantages, returns = self.compute_advantages(rewards, values)
        
        # 执行PPO更新
        self.ppo_update(sequences, old_log_probs, torch.tensor(rewards, device=self.device), advantages)
        
        # 更新统计信息
        if len(rewards) > 0:
            self.stats['mean_reward'].append(np.mean(rewards))
            self.stats['mean_te'].append(np.mean(te_values))
            self.stats['mean_length'].append(np.mean([len(seq) for seq in sequences]))
            
            if len(rewards) > 0:
                best_idx = np.argmax(te_values)
                if best_idx < len(decoded_sequences):
                    print(f"本批次最佳序列 (TE={te_values[best_idx]:.4f}):\n{decoded_sequences[best_idx][:100]}")
        
        return True
    
    def train(self):
        """完整的训练循环"""
        # 保存训练配置
        self._save_config()
        
        for iteration in tqdm(range(self.config.max_iterations)):
            success = self.train_iteration()
            
            if not success:
                print(f"迭代 {iteration} 失败，跳过")
                self._log(f"迭代 {iteration} 失败，跳过")
                continue
            
            # 每10次迭代输出日志
            if (iteration + 1) % 10 == 0:
                log_msg = f"迭代 {iteration+1}: 平均奖励 = {self.stats['mean_reward'][-1]:.4f}, " \
                         f"平均TE = {self.stats['mean_te'][-1]:.4f}, " \
                         f"平均长度 = {self.stats['mean_length'][-1]:.1f}"
                print(log_msg)
                self._log(log_msg)
            
            # 保存检查点
            if (iteration + 1) % self.config.save_interval == 0:
                # 保存模型
                model_filename = f"model_iter_{iteration+1}"
                self.save_model(model_filename)
                
                # 保存高TE序列
                if len(self.sequence_storage) > 0:
                    seq_filename = f"high_te_sequences_iter_{iteration+1}"
                    self.save_sequences(seq_filename)
                
                # 保存训练统计信息
                self._save_stats(f"stats_iter_{iteration+1}.json")
        
        # 训练结束时保存最终模型和序列
        self.save_model("final_model")
        if len(self.sequence_storage) > 0:
            self.save_sequences("final_high_te_sequences")
        self._save_stats("final_stats.json")
    
    def _log(self, message):
        """写入日志文件"""
        with open(self.log_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    
    def _save_config(self):
        """保存训练配置"""
        config_path = os.path.join(self.output_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(vars(self.config), f)
    
    def _save_stats(self, filename):
        """保存训练统计信息"""
        import json
        
        # 将numpy数组转换为列表
        stats_to_save = {
            'mean_reward': [float(x) for x in self.stats['mean_reward']],
            'mean_te': [float(x) for x in self.stats['mean_te']],
            'mean_length': [float(x) for x in self.stats['mean_length']],
            'max_te': float(self.stats['max_te']),
            'best_sequence': self.stats['best_sequence']
        }
        
        stats_path = os.path.join(self.log_dir, filename)
        with open(stats_path, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
    
    def save_model(self, filename):
        """保存模型"""
        model_path = os.path.join(self.model_dir, f"{filename}.pt")
        torch.save({
            'actor_state_dict': self.actor_model.state_dict(),
            'critic_state_dict': self.ac_model.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': self.stats,
            'config': self.config
        }, model_path)
        self._log(f"模型已保存到 {model_path}")
    
    def save_sequences(self, filename):
        """保存生成的高TE序列"""
        # 将buffer转换为更易读的格式：(序列, 解码序列, TE值)
        sequences_to_save = []
        for seq, decoded_seq, te_score in self.sequence_storage.buffer:
            sequences_to_save.append({
                'codon_sequence': seq,
                'decoded_sequence': decoded_seq,
                'te_score': float(te_score)
            })
            
        # 按TE值排序
        sequences_to_save.sort(key=lambda x: x['te_score'], reverse=True)
        
        # 保存为pickle格式
        pkl_path = os.path.join(self.sequence_dir, f"{filename}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(sequences_to_save, f)
        
        # 保存为文本格式，更易于阅读
        txt_path = os.path.join(self.sequence_dir, f"{filename}.txt")
        with open(txt_path, 'w') as f:
            f.write(f"总共保存了 {len(sequences_to_save)} 个高TE序列\n\n")
            for i, item in enumerate(sequences_to_save[:20]):
                f.write(f"序列 {i+1}, TE值: {item['te_score']:.4f}\n")
                f.write(f"{item['decoded_sequence']}\n\n")
        
        self._log(f"高TE序列已保存到 {pkl_path} 和 {txt_path}")


def load_te_model():
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

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='PPO微调mRNA设计器')
    parser.add_argument('--output_dir', type=str, default='ppo_output', help='输出目录')
    parser.add_argument('--max_iterations', type=int, default=1000, help='最大训练迭代次数')
    parser.add_argument('--save_interval', type=int, default=10, help='保存模型的迭代间隔')
    parser.add_argument('--generate_batch_size', type=int, default=16, help='生成序列的批次大小')
    parser.add_argument('--checkpoint', type=str, default="output/ckpt_563000.pt", help='预训练模型检查点路径')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    VOCAB_FILE = "tokenizer/vocab.txt"
    tokenizer = BertTokenizerFast(vocab_file=VOCAB_FILE, do_lower_case=False)
    
    # 加载预训练模型
    ckpt_path = args.checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    gptconf = GPTConfig(**checkpoint['model_args'])
    actor_model = GPT(gptconf).to(device)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    actor_model.load_state_dict(state_dict)
    
    # 加载TE预测模型
    reward_model = load_te_model()
    
    # 创建PPO配置
    ppo_config = PPOConfig()
    # 从命令行参数更新配置
    ppo_config.output_dir = args.output_dir
    ppo_config.max_iterations = args.max_iterations
    ppo_config.save_interval = args.save_interval
    ppo_config.generate_batch_size = args.generate_batch_size
    
    # 创建并开始训练
    trainer = PPOTrainer(ppo_config, actor_model, reward_model, tokenizer, device)
    
    trainer.train()
    
    # 记录训练结束信息
    trainer._log(f"\n训练完成!")
    trainer._log(f"最佳序列的TE值: {trainer.stats['max_te']:.4f}")
    trainer._log(f"最佳序列: {trainer.stats['best_sequence']}")
    
    print(f"\n训练完成!")
    print(f"最佳序列的TE值: {trainer.stats['max_te']:.4f}")
    print(f"最佳序列: {trainer.stats['best_sequence']}")
    print(f"所有输出已保存到 {trainer.output_dir} 目录")
    
    # 生成最终测试序列
    print("\n生成最终测试序列...")
    try:
        _, final_sequences = trainer.generate_sequences(10)
        
        # 记录生成的测试序列
        test_seq_path = os.path.join(trainer.sequence_dir, "final_test_sequences.txt")
        with open(test_seq_path, 'w') as f:
            for i, seq in enumerate(final_sequences):
                print(f"测试序列 {i+1}:\n{seq[:100]}" + ("..." if len(seq) > 100 else ""))
                f.write(f"测试序列 {i+1}:\n{seq}\n\n")
            
            rewards, te_values = trainer.calculate_reward(final_sequences)
            for i, (seq, te) in enumerate(zip(final_sequences, te_values)):
                print(f"测试序列 {i+1} TE值: {te:.4f}")
                f.write(f"测试序列 {i+1} TE值: {te:.4f}\n\n")
        
        trainer._log(f"最终测试序列已保存到 {test_seq_path}")
    except Exception as e:
        error_msg = f"生成测试序列时出错: {e}"
        print(error_msg)
        trainer._log(error_msg)

if __name__ == "__main__":
    main() 


# CUDA_VISIBLE_DEVICES=0 python ppo_finetune.py --output_dir ppo_runs --max_iterations 500 --save_interval 20
# CUDA_VISIBLE_DEVICES=0 python ppo_finetune.py --output_dir ppo_runs --max_iterations 500 --save_interval 20