"""
To finetune a GPT model with SFT (Supervised Fine-Tuning)
"""
import argparse
import os
import time
import math
import pickle
import pynvml
import yaml
import pandas as pd
import psutil
from tqdm import tqdm
import random

from contextlib import nullcontext
from transformers import AutoTokenizer, BertTokenizerFast
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

def log_and_write(filename, message):
    with open(filename, 'a') as f:
        f.write(message + "\n")
    print(message)


def print_memory_usage():
    process = psutil.Process()
    memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    print(f"Memory usage: {memory:.2f} MB")

class ExpressionDataset(Dataset):
    def __init__(self, csv_path, block_size, tokenizer, split='train'):
        """
        Initialize the expression dataset from CSV.
        
        Args:
            csv_path (str): Path to the CSV file.
            block_size (int): Maximum token length for each sequence.
            tokenizer: Tokenizer to convert sequences to token IDs.
            split (str): 'train' or 'val' split.
        """
        self.block_size = block_size
        self.tokenizer = tokenizer
        
        # Load data from CSV
        df = pd.read_csv(csv_path)
        
        # Filter by split
        df = df[df['Split'] == split]
        
        # Extract sequences and values
        self.sequences = df['Sequence'].tolist()
        self.values = df['Value'].tolist()
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Fetch a single data point.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            dict: Contains tokenized sequence and target value.
        """
        sequence = self.sequences[idx]
        value = self.values[idx]
        
        # 添加[SEP]标记在序列前后，匹配预训练数据格式
        formatted_sequence = "[SEP]" + sequence + "[SEP]"
        
        # Tokenize sequence
        tokens = self.tokenizer.encode(formatted_sequence)
        
        # Truncate or pad to block_size
        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]
        else:
            tokens = tokens + [0] * (self.block_size - len(tokens))
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        label_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "label_ids": label_ids,
            "value": torch.tensor(value, dtype=torch.float)
        }

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    def __init__(self, patience=5, min_delta=0, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            verbose (bool): If True, prints a message for each improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Load configuration file")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()

def main():
    args = parse_arguments()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)

    VOCAB_FILE = "tokenizer/vocab.txt"
    tokenizer = BertTokenizerFast(vocab_file=VOCAB_FILE, do_lower_case=False)

    print(f'tokenizer length:{len(tokenizer)}')
    # -----------------------------------------------------------------------------
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=config['backend'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank
        assert config['gradient_accumulation_steps'] % ddp_world_size == 0
        config['gradient_accumulation_steps'] //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    if master_process:
        os.makedirs(config['out_dir'], exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in config['device'] else 'cpu' # for later use in torch.autocast

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Initialize the dataset
    csv_path = config['data_path']
    train_dataset = ExpressionDataset(csv_path, config['block_size'], tokenizer, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_dataset = ExpressionDataset(csv_path, config['block_size'], tokenizer, 'val')                      
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    # 计算每个epoch的步数
    steps_per_epoch = len(train_dataloader)
    log_and_write(config['log_dir'], f"数据集大小: {len(train_dataset)}个训练样本, {len(val_dataset)}个验证样本")
    log_and_write(config['log_dir'], f"每个epoch包含 {steps_per_epoch} 步")
    
    # 如果配置中指定了epochs，则计算对应的最大迭代次数
    if 'epochs' in config:
        config['max_iters'] = config['epochs'] * steps_per_epoch
        log_and_write(config['log_dir'], f"训练 {config['epochs']} 个epochs，总共 {config['max_iters']} 步")

    iter_num = 0
    best_val_loss = 1e9
    current_epoch = 0
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience', 5),
        min_delta=config.get('early_stopping_min_delta', 0.01),
        verbose=True
    )

    # model init
    model_args = dict(n_layer=config['n_layer'], n_head=config['n_head'], n_embd=config['n_embd'], block_size=config['block_size'], bias=config['bias'], vocab_size=None, dropout=config['dropout']) # start with model_args from command line
    if config['init_from'] == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if config['meta_vocab_size'] is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = config['meta_vocab_size'] if config['meta_vocab_size'] is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif config['init_from'] == 'resume':
        print(f"Resuming training from {config['out_dir']}")
        # resume training from a checkpoint.
        checkpoint = torch.load(config['ckpt_path'], map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        # 计算当前epoch
        current_epoch = 0

    # crop down the model block size if desired, using model surgery
    if config['block_size'] < model.config.block_size:
        model.crop_block_size(config['block_size'])
        model_args['block_size'] = config['block_size'] # so that the checkpoint will have the right value
    model.to(config['device'])

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(config['dtype'] == 'float32'))

    # optimizer
    optimizer = model.configure_optimizers(config['weight_decay'], config['learning_rate'], (config['beta1'], config['beta2']), device_type)
    if config['init_from'] == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if config.get('compile', False):
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
    
    # wrap model into DDP container
    if ddp:
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank], find_unused_parameters=False)

    pynvml.nvmlInit()
    tokens_per_iter = config['gradient_accumulation_steps'] * ddp_world_size * config['batch_size'] * config['block_size']
    print('ddp_world_size:',ddp_world_size)
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    def print_gpu_memory_usage():
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"Used: {info.used / 1024**2:.2f}MB/{info.total / 1024**2:.2f}MB ({info.used / info.total * 100:.2f}%)")

    @torch.no_grad()
    def estimate_loss():
        out = {}
        perplexities = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(config['eval_iters'])
            total_loss = 0
            dataloader = train_dataloader if split == 'train' else val_dataloader
            dataloader_iter = iter(dataloader)
            dataset_size = len(dataloader.dataset) 
            batch_size = dataloader.batch_size

            for k in range(config['eval_iters']):
                try:
                    if dataset_size > batch_size:
                        random_index = random.randint(0, dataset_size - batch_size)
                    else:
                        random_index = 0
                
                    batch = next(iter(torch.utils.data.DataLoader(
                        dataloader.dataset, batch_size=batch_size, shuffle=False,
                        sampler=torch.utils.data.SubsetRandomSampler(range(random_index, random_index + batch_size))
                    )))
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)
                
                X = batch["input_ids"].to("cuda" if torch.cuda.is_available() else "cpu")
                Y = batch["label_ids"].to("cuda" if torch.cuda.is_available() else "cpu")
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
                total_loss += loss.item()
            avg_loss = losses.mean()
            out[split] = avg_loss
            perplexities[split] = torch.exp(avg_loss)
        model.train()
        return out, perplexities
    
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < config['warmup_iters']:
            return config['learning_rate'] * it / config['warmup_iters']
        # 2) if it > lr_decay_iters, return min learning rate
        if it > config['lr_decay_iters']:
            return config['min_lr']
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])
    
    # training loop
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0
    
    while True:
        # 检查是否达到最大epoch数
        if 'epochs' in config and current_epoch >= config['epochs']:
            log_and_write(config['log_dir'], f"已完成 {config['epochs']} 个epochs的训练")
            break
            
        # 检查是否达到最大迭代次数
        if iter_num >= config['max_iters']:
            log_and_write(config['log_dir'], f"已达到最大迭代次数 {config['max_iters']}")
            break
            
        log_and_write(config['log_dir'], f"开始 Epoch {current_epoch + 1}")
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move input and labels to GPU
            X = batch["input_ids"].to("cuda" if torch.cuda.is_available() else "cpu")
            Y = batch["label_ids"].to("cuda" if torch.cuda.is_available() else "cpu")

            # Determine and set the learning rate for this iteration
            lr = get_lr(iter_num) if config['decay_lr'] else config['learning_rate']
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Evaluate the loss on train/val sets and save checkpoints periodically
            if iter_num % config['eval_interval'] == 0 and master_process:
                losses, perplexities = estimate_loss()
                log_and_write(config['log_dir'], f"step {iter_num} (epoch {current_epoch + 1}): train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, train perplexity: {perplexities['train']:.4f}, val perplexity: {perplexities['val']:.4f}")

                if iter_num % 200 == 0:
                    print_gpu_memory_usage()

                # Check early stopping
                is_improved = early_stopping(losses['val'])
                
                if config['always_save_checkpoint'] and is_improved:
                    best_val_loss = losses['val']
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'epoch': current_epoch,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    log_and_write(config['log_dir'], f"saving checkpoint to {config['out_dir']}")
                    torch.save(checkpoint, os.path.join(config['out_dir'], f'ckpt_epoch{current_epoch+1}_iter{iter_num}.pt'))
                    # Also save best model
                    torch.save(checkpoint, os.path.join(config['out_dir'], 'best_model.pt'))
                
                # Early stopping
                if early_stopping.early_stop:
                    log_and_write(config['log_dir'], f"Early stopping triggered after {iter_num} iterations (epoch {current_epoch+1})")
                    if ddp:
                        destroy_process_group()
                    pynvml.nvmlShutdown()
                    return

            if iter_num == 0 and config['eval_only']:
                break

            # Forward, backward, update (gradient accumulation handled here)
            for micro_step in range(config['gradient_accumulation_steps']):
                if ddp:
                    model.require_backward_grad_sync = (micro_step == config['gradient_accumulation_steps'] - 1)

                with torch.amp.autocast(device_type='cuda', enabled=True):
                    logits, loss = model(X, Y)
                    loss = loss / config['gradient_accumulation_steps']  # Scale the loss for gradient accumulation

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

            # Gradient clipping
            if config['grad_clip'] != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

            # Optimizer step and scaler update
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients to release memory
            optimizer.zero_grad(set_to_none=True)

            # Timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % config['log_interval'] == 0 and master_process:
                lossf = loss.item() * config['gradient_accumulation_steps']  # Undo scaling for logging
                if local_iter_num >= 5:  # Allow the training loop to stabilize
                    mfu = raw_model.estimate_mfu(config['batch_size'] * config['gradient_accumulation_steps'], dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                log_and_write(config['log_dir'], f"iter {iter_num} (epoch {current_epoch+1}): loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr}, mfu {running_mfu*100:.2f}%")

            iter_num += 1
            local_iter_num += 1

            if iter_num >= config['max_iters']:
                break
                
        # 一个epoch结束
        epoch_time = time.time() - epoch_start_time
        current_epoch += 1
        
        # 在每个epoch结束时评估和保存
        if master_process:
            losses, perplexities = estimate_loss()
            log_and_write(config['log_dir'], f"Epoch {current_epoch} 完成: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, train perplexity: {perplexities['train']:.4f}, val perplexity: {perplexities['val']:.4f}, 用时 {epoch_time:.2f}秒")
            
            # 每个epoch结束时保存检查点
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'epoch': current_epoch,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            torch.save(checkpoint, os.path.join(config['out_dir'], f'ckpt_epoch{current_epoch}.pt'))
            
            # 检查early stopping
            is_improved = early_stopping(losses['val'])
            if is_improved:
                best_val_loss = losses['val']
                torch.save(checkpoint, os.path.join(config['out_dir'], 'best_model.pt'))
                
            if early_stopping.early_stop:
                log_and_write(config['log_dir'], f"Early stopping triggered after epoch {current_epoch}")
                break
    
    log_and_write(config['log_dir'], f"训练完成，共运行 {current_epoch} 个epochs，{iter_num} 步")
    
    if ddp:
        destroy_process_group()

    pynvml.nvmlShutdown()
    
if __name__ == "__main__":
    main()


# python sft_exp.py --config ./configs/sft.yaml
# CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node=2 sft_exp.py --config ./configs/sft_exp.yaml 