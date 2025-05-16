"""
SFT (Supervised Fine-Tuning) 代码，包含序列生成和回归任务
"""
import argparse
import os
import time
import math
import yaml
import lmdb
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
from contextlib import nullcontext
from transformers import BertTokenizerFast

from model import GPTConfig, GPT

def log_and_write(filename, message):
    with open(filename, 'a') as f:
        f.write(message + "\n")
    print(message)

class RNARegDataset(Dataset):
    """RNA序列和TE值的数据集，用于回归任务"""
    def __init__(self, csv_path, block_size, tokenizer):
        """
        初始化RNA回归数据集
        
        Args:
            csv_path (str): CSV文件路径
            block_size (int): 序列最大长度
            tokenizer: 分词器
        """
        self.block_size = block_size
        self.tokenizer = tokenizer
        
        # 读取CSV数据
        self.data = pd.read_csv(csv_path)
        print(f"Loaded {len(self.data)} sequences from {csv_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取一个数据样本"""
        # 获取序列和TE值
        row = self.data.iloc[idx]
        sequence = row['CDS_sequence']
        te_value = row['TE']
        
        rna_sequence = sequence.replace("T", "U")
        
        # 检查序列是否合法（只包含A、U、C、G且长度是3的倍数）
        if not all(base in "AUCG" for base in rna_sequence) or len(rna_sequence) % 3 != 0:
            print(f"警告：序列 {idx} 不合法，包含非法碱基或长度不是3的倍数")
        
        # 将序列转换为用空格分隔的密码子形式
        codon_sequence = ' '.join(rna_sequence[i:i+3] for i in range(0, len(rna_sequence), 3))
        
        # 使用处理后的密码子序列
        sequence = codon_sequence
        formatted_sequence = "[SEP]" + sequence + "[SEP]"
        
        # 使用tokenizer直接编码整个序列
        tokens = self.tokenizer.encode(formatted_sequence)

        # 截断或填充到block_size
        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]
        else:
            tokens = tokens + [0] * (self.block_size - len(tokens))
        
        # 转换为张量
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        label_ids = torch.tensor(tokens[1:], dtype=torch.long)
        te_value = torch.tensor(te_value, dtype=torch.float)
        
        return {
            "input_ids": input_ids,
            "label_ids": label_ids,
            "te_value": te_value
        }

class GPTTE(GPT):
    """扩展GPT模型，增加回归头以预测TE值"""
    def __init__(self, config):
        super().__init__(config)
        # 添加回归头
        self.te_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, 1)
        )
    
    def forward(self, idx, targets=None, te_targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # GPT模型前向传播
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            # lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            lm_loss = None
        else:
            logits = self.lm_head(x[:, [-1], :])
            lm_loss = None
        
        # 计算回归损失 (使用序列的平均表示)
        te_pred = None
        reg_loss = None
        if te_targets is not None:
            # 使用序列的最后一个隐藏状态作为整个序列的表示
            seq_repr = x[:, -1, :]
            te_pred = self.te_head(seq_repr).squeeze(-1)
            reg_loss = F.mse_loss(te_pred, te_targets)
        
        # 总损失是语言模型损失和回归损失的加权和
        loss = None
        if lm_loss is not None and reg_loss is not None:
            loss = lm_loss + reg_loss
        elif lm_loss is not None:
            loss = lm_loss
        elif reg_loss is not None:
            loss = reg_loss
            
        return logits, loss, te_pred, lm_loss, reg_loss

def parse_arguments():
    parser = argparse.ArgumentParser(description="Load configuration file")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()

def main():
    args = parse_arguments()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)

    # 加载分词器
    VOCAB_FILE = "tokenizer/vocab.txt"
    tokenizer = BertTokenizerFast(vocab_file=VOCAB_FILE, do_lower_case=False)
    print(f'tokenizer length:{len(tokenizer)}')
    
    # 分布式训练设置
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=config['backend'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert config['gradient_accumulation_steps'] % ddp_world_size == 0
        config['gradient_accumulation_steps'] //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = config['device']

    tokens_per_iter = config['gradient_accumulation_steps'] * ddp_world_size * config['batch_size'] * config['block_size']
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(config['out_dir'], exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # 加载数据集
    data_path = config['data_path']
    df = pd.read_csv(data_path)
    
    # 分割数据集
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    # 保存训练和验证集
    train_csv_path = os.path.join(os.path.dirname(data_path), "train_sft.csv")
    val_csv_path = os.path.join(os.path.dirname(data_path), "val_sft.csv")
    train_df.to_csv(train_csv_path)
    val_df.to_csv(val_csv_path)
    
    train_dataset = RNARegDataset(train_csv_path, config['block_size'], tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_dataset = RNARegDataset(val_csv_path, config['block_size'], tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    best_val_loss = 1e9
    start_epoch = 0

    model_args = dict(
        n_layer=config['n_layer'], 
        n_head=config['n_head'], 
        n_embd=config['n_embd'], 
        block_size=config['block_size'], 
        bias=config['bias'], 
        vocab_size=None, 
        dropout=config['dropout']
    )
    
    if config['init_from'] == 'scratch':
        print("Initializing a new model from scratch")
        model_args['vocab_size'] = config['meta_vocab_size'] if config['meta_vocab_size'] is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPTTE(gptconf)
    else:
        ckpt_path = config.get('ckpt_path', '')
        if not ckpt_path and config['init_from'].startswith('pretrained:'):
            ckpt_path = config['init_from'].split(':')[1]
        
        print(f"Loading model from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # 获取模型参数
        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        
        # 创建模型
        gptconf = GPTConfig(**model_args)
        model = GPTTE(gptconf)
        
        # 加载模型权重
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict, strict=False)
        
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', 1e9)
            print(f"Resuming from epoch {start_epoch}")

    if config['block_size'] < model.config.block_size:
        model.crop_block_size(config['block_size'])
        model_args['block_size'] = config['block_size']
    
    model.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(config['dtype'] == 'float32'))

    # 优化器
    optimizer = model.configure_optimizers(config['weight_decay'], config['learning_rate'], (config['beta1'], config['beta2']), device_type)
    if config['init_from'] != 'scratch' and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded optimizer state")
        except:
            print("Could not load optimizer state, initializing fresh")
    
    if config.get('compile', False) and hasattr(torch, 'compile'):
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)
    
    if ddp:
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank], find_unused_parameters=False)

    @torch.no_grad()
    def evaluate():
        out = {}
        perplexities = {}
        te_mses = {}
        model.eval()
        for split in ['train', 'val']:
            dataloader = train_dataloader if split == 'train' else val_dataloader
            
            total_loss = 0
            total_lm_loss = 0
            total_reg_loss = 0
            num_batches = 0
            
            for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
                X = batch["input_ids"].to(device)
                Y = batch["label_ids"].to(device)
                TE = batch["te_value"].to(device)
                
                with ctx:
                    logits, loss, te_pred, lm_loss, reg_loss = model(X, Y, TE)
                
                total_loss += loss.item()
                total_lm_loss += lm_loss.item() if lm_loss is not None else 0
                total_reg_loss += reg_loss.item() if reg_loss is not None else 0
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            avg_lm_loss = total_lm_loss / num_batches
            avg_reg_loss = total_reg_loss / num_batches
            
            out[split] = avg_loss
            perplexities[split] = torch.exp(torch.tensor(avg_lm_loss))
            te_mses[split] = avg_reg_loss
            
        model.train()
        return out, perplexities, te_mses

    def get_lr(epoch, iter_in_epoch, total_iters_per_epoch):
        total_iters = epoch * total_iters_per_epoch + iter_in_epoch
        
        # 根据总迭代次数计算学习率
        if total_iters < config['warmup_iters']:
            return config['learning_rate'] * total_iters / config['warmup_iters']
        
        # 计算总训练迭代次数
        max_iters = config['epochs'] * total_iters_per_epoch
        
        if total_iters > max_iters * 0.9:  # 在最后10%的迭代中使用最小学习率
            return config['min_lr']
        
        # 余弦退火学习率
        decay_ratio = (total_iters - config['warmup_iters']) / (max_iters * 0.9 - config['warmup_iters'])
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])

    # 训练循环 - 按照epoch训练
    num_epochs = config.get('epochs', 10)
    t0 = time.time()
    raw_model = model.module if ddp else model
    running_mfu = -1.0
    
    # 计算每个epoch的总迭代次数
    total_iters_per_epoch = len(train_dataloader)
    
    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        epoch_lm_loss = 0
        epoch_reg_loss = 0
        
        # 使用tqdm显示进度条
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                    desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, batch in pbar:
            # 将输入和标签移至GPU
            X = batch["input_ids"].to(device)
            Y = batch["label_ids"].to(device)
            TE = batch["te_value"].to(device)

            # 设置学习率
            lr = get_lr(epoch, i, total_iters_per_epoch) if config['decay_lr'] else config['learning_rate']
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # 前向、后向、更新（梯度累积）
            for micro_step in range(config['gradient_accumulation_steps']):
                if ddp:
                    model.require_backward_grad_sync = (micro_step == config['gradient_accumulation_steps'] - 1)

                with ctx:
                    logits, loss, te_pred, lm_loss, reg_loss = model(X, Y, TE)
                    loss = loss / config['gradient_accumulation_steps']

                # 反向传播
                scaler.scale(loss).backward()

            # 梯度裁剪
            if config['grad_clip'] != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

            # 优化器步骤
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # 累计损失
            epoch_loss += loss.item() * config['gradient_accumulation_steps']
            epoch_lm_loss += lm_loss.item() if lm_loss is not None else 0
            epoch_reg_loss += reg_loss.item() if reg_loss is not None else 0
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{epoch_loss/(i+1):.4f}", 
                'lm_loss': f"{epoch_lm_loss/(i+1):.4f}", 
                'reg_loss': f"{epoch_reg_loss/(i+1):.4f}", 
                'lr': f"{lr:.6f}"
            })
            
            # 计算MFU（如果需要）
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if i >= 5:  # 跳过前几个批次
                mfu = raw_model.estimate_mfu(config['batch_size'] * config['gradient_accumulation_steps'], dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        
        # 每个epoch结束后评估模型
        if master_process:
            # 计算平均损失
            avg_loss = epoch_loss / len(train_dataloader)
            avg_lm_loss = epoch_lm_loss / len(train_dataloader)
            avg_reg_loss = epoch_reg_loss / len(train_dataloader)
            
            log_and_write(config['log_dir'], 
                          f"Epoch {epoch+1}/{num_epochs}: train loss {avg_loss:.4f}, "
                          f"lm_loss {avg_lm_loss:.4f}, reg_loss {avg_reg_loss:.4f}, "
                          f"mfu {running_mfu*100:.2f}%")
            
            # 评估验证集
            losses, perplexities, te_mses = evaluate()
            log_and_write(config['log_dir'], 
                          f"Epoch {epoch+1}: val loss {losses['val']:.4f}, "
                          f"val ppl: {perplexities['val']:.4f}, "
                          f"val TE MSE: {te_mses['val']:.4f}")
            
            # 保存最佳模型
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': raw_model.state_dict()
                }
                log_and_write(config['log_dir'], f"saving checkpoint to {config['out_dir']}")
                torch.save(checkpoint, os.path.join(config['out_dir'], f'sft_ckpt_best.pt'))
            
            checkpoint = {
                'model': raw_model.state_dict()
            }
            torch.save(checkpoint, os.path.join(config['out_dir'], f'sft_ckpt_latest.pt'))
    
    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()

# 运行命令示例:
# python sft.py --config ./configs/sft.yaml
# 分布式训练:
# CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node=2 sft.py --config ./configs/sft_te.yaml 
# CUDA_VISIBLE_DEVICES=3 python sft_te.py --config ./configs/sft_te.yaml