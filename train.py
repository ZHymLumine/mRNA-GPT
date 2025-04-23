"""
To train a GPT from sratch
"""
import argparse
import os
import time
import math
import pickle
import pynvml
import yaml
import lmdb
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
from data.alphabet import Alphabet

def log_and_write(filename, message):
    with open(filename, 'a') as f:
        f.write(message + "\n")
    print(message)


def print_memory_usage():
    process = psutil.Process()
    memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    print(f"Memory usage: {memory:.2f} MB")

class RNADataset(Dataset):
    def __init__(self, lmdb_path, block_size, tokenizer):
        """
        Initialize the LMDB dataset.
        
        Args:
            lmdb_path (str): Path to the LMDB file.
            block_size (int): Maximum token length for each sequence.
        """
        self.lmdb_path = lmdb_path
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.env.begin() as txn:
            self.total_samples = txn.stat()["entries"]
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """
        Fetch a single data point from the LMDB file.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            np.ndarray: Tokenized sequence (padded/truncated to block_size).
        """
        with self.env.begin() as txn:
            key = f"{idx}".encode("utf-8")
            value = txn.get(key)
            if value is None:
                raise IndexError(f"Index {idx} not found in LMDB.")
            
            # Decode the value
            data = np.frombuffer(value, dtype=np.int32)

            # Truncate or pad to block_size
            if len(data) > self.block_size:
                data = data[:self.block_size]
            else:
                data = np.pad(data, (0, self.block_size - len(data)), constant_values=0)

            input_ids = torch.tensor(data[:-1], dtype=torch.long)
            label_ids = torch.tensor(data[1:], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "label_ids": label_ids,
            }
    
    def close(self):
        self.env.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Load configuration file")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()

def main():
    args = parse_arguments()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)

    tokenizer = AutoTokenizer.from_pretrained("ZYMScott/mRNAdesigner_codon")
    # tokenizer = Alphabet.load("../tokenizer_codon.json")
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

    tokens_per_iter = config['gradient_accumulation_steps'] * ddp_world_size * config['batch_size'] * config['block_size']
    print('ddp_world_size:',ddp_world_size)
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    pynvml.nvmlInit()
    def print_gpu_memory_usage():
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"Used: {info.used / 1024**2:.2f}MB/{info.total / 1024**2:.2f}MB ({info.used / info.total * 100:.2f}%)")

    if master_process:
        os.makedirs(config['out_dir'], exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in config['device'] else 'cpu' # for later use in torch.autocast

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Initialize the dataset
    train_dataset = RNADataset(os.path.join(config['data_dir'], 'train_codon.lmdb'), config['block_size'], tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_dataset = RNADataset(os.path.join(config['data_dir'], 'val_codon.lmdb'), config['block_size'], tokenizer)                      
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    iter_num = 0
    best_val_loss = 1e9

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
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

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
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
    
    # wrap model into DDP container
    if ddp:
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

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

    while iter_num < config['max_iters']:
        for batch in train_dataloader:
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
                log_and_write(config['log_dir'], f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, train perplexity: {perplexities['train']:.4f}, val perplexity: {perplexities['val']:.4f}")

                if iter_num % 200 == 0:
                    print_gpu_memory_usage()

                if config['always_save_checkpoint'] and losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    log_and_write(config['log_dir'], f"saving checkpoint to {config['out_dir']}")
                    torch.save(checkpoint, os.path.join(config['out_dir'], f'ckpt_{iter_num}.pt'))

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
                log_and_write(config['log_dir'], f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr}, mfu {running_mfu*100:.2f}%")

            iter_num += 1
            local_iter_num += 1

            if iter_num > config['max_iters']:
                break
    
    if ddp:
        destroy_process_group()

    pynvml.nvmlShutdown()
    
if __name__ == "__main__":
    main()


# python train.py --config ./configs/pretrain.yaml
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --config ./configs/pretrain.yaml
# Node 0
# MASTER_ADDR="node0_ip" MASTER_PORT=12345 RANK=0 WORLD_SIZE=8 torchrun --nproc_per_node=4 train.py --config ./configs/pretrain.yaml
# Node 1
# MASTER_ADDR="node0_ip" MASTER_PORT=12345 RANK=1 WORLD_SIZE=8 torchrun --nproc_per_node=4 train.py --config ./configs/pretrain.yaml
"""
To train a GPT from sratch
"""
import argparse
import os
import time
import math
import pickle
import pynvml
import yaml
import lmdb
import psutil
from tqdm import tqdm
import random

from contextlib import nullcontext
from transformers import AutoTokenizer
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from data.alphabet import Alphabet

def log_and_write(filename, message):
    with open(filename, 'a') as f:
        f.write(message + "\n")
    print(message)


def print_memory_usage():
    process = psutil.Process()
    memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    print(f"Memory usage: {memory:.2f} MB")

class RNADataset(Dataset):
    def __init__(self, lmdb_path, block_size, tokenizer):
        """
        Initialize the LMDB dataset.
        
        Args:
            lmdb_path (str): Path to the LMDB file.
            block_size (int): Maximum token length for each sequence.
        """
        self.lmdb_path = lmdb_path
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.env.begin() as txn:
            self.total_samples = txn.stat()["entries"]
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """
        Fetch a single data point from the LMDB file.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            np.ndarray: Tokenized sequence (padded/truncated to block_size).
        """
        with self.env.begin() as txn:
            key = f"{idx}".encode("utf-8")
            value = txn.get(key)
            if value is None:
                raise IndexError(f"Index {idx} not found in LMDB.")
            
            # Decode the value
            data = np.frombuffer(value, dtype=np.int32)

            # Truncate or pad to block_size
            if len(data) > self.block_size:
                data = data[:self.block_size]
            else:
                data = np.pad(data, (0, self.block_size - len(data)), constant_values=0)

            input_ids = torch.tensor(data[:-1], dtype=torch.long)
            label_ids = torch.tensor(data[1:], dtype=torch.long)

            return {
                "input_ids": input_ids,
                "label_ids": label_ids,
            }
    
    def close(self):
        self.env.close()

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
    # tokenizer = Alphabet.load("../tokenizer_codon.json")
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

    tokens_per_iter = config['gradient_accumulation_steps'] * ddp_world_size * config['batch_size'] * config['block_size']
    print('ddp_world_size:',ddp_world_size)
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    pynvml.nvmlInit()
    def print_gpu_memory_usage():
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"Used: {info.used / 1024**2:.2f}MB/{info.total / 1024**2:.2f}MB ({info.used / info.total * 100:.2f}%)")

    if master_process:
        os.makedirs(config['out_dir'], exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in config['device'] else 'cpu' # for later use in torch.autocast

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Initialize the dataset
    train_dataset = RNADataset(os.path.join(config['data_dir'], 'train_codon.lmdb'), config['block_size'], tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_dataset = RNADataset(os.path.join(config['data_dir'], 'val_codon.lmdb'), config['block_size'], tokenizer)                      
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    iter_num = 0
    best_val_loss = 1e9

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
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

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
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
    
    # wrap model into DDP container
    if ddp:
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

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

    while iter_num < config['max_iters']:
        for batch in train_dataloader:
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
                log_and_write(config['log_dir'], f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, train perplexity: {perplexities['train']:.4f}, val perplexity: {perplexities['val']:.4f}")

                if iter_num % 200 == 0:
                    print_gpu_memory_usage()

                if config['always_save_checkpoint'] and losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    log_and_write(config['log_dir'], f"saving checkpoint to {config['out_dir']}")
                    torch.save(checkpoint, os.path.join(config['out_dir'], f'ckpt_{iter_num}.pt'))

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
                log_and_write(config['log_dir'], f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr}, mfu {running_mfu*100:.2f}%")

            iter_num += 1
            local_iter_num += 1

            if iter_num > config['max_iters']:
                break
    
    if ddp:
        destroy_process_group()

    pynvml.nvmlShutdown()
    
if __name__ == "__main__":
    main()


# python train.py --config ./configs/pretrain.yaml
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --config ./configs/pretrain.yaml
# Node 0
# MASTER_ADDR="node0_ip" MASTER_PORT=12345 RANK=0 WORLD_SIZE=8 torchrun --nproc_per_node=4 train.py --config ./configs/pretrain.yaml
# Node 1
# MASTER_ADDR="node0_ip" MASTER_PORT=12345 RANK=1 WORLD_SIZE=8 torchrun --nproc_per_node=4 train.py --config ./configs/pretrain.yaml
