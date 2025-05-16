#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize training and validation losses
"""

import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def parse_log_file(log_file):
    """Parse log file to extract training and validation losses"""
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Extract training losses
    train_pattern = r"Epoch (\d+)/\d+: train loss ([0-9.]+), lm_loss ([0-9.]+), reg_loss ([0-9.]+)"
    train_matches = re.findall(train_pattern, log_content)
    
    # Extract validation losses
    val_pattern = r"Epoch (\d+): val loss ([0-9.]+), val ppl: ([0-9.]+), val TE MSE: ([0-9.]+)"
    val_matches = re.findall(val_pattern, log_content)
    
    # Organize data
    epochs = []
    train_losses = []
    train_lm_losses = []
    train_reg_losses = []
    val_losses = []
    val_ppls = []
    val_te_mses = []
    
    for match in train_matches:
        epoch = int(match[0])
        train_loss = float(match[1])
        train_lm_loss = float(match[2])
        train_reg_loss = float(match[3])
        
        epochs.append(epoch)
        train_losses.append(train_loss)
        train_lm_losses.append(train_lm_loss)
        train_reg_losses.append(train_reg_loss)
    
    val_epochs = []
    for match in val_matches:
        epoch = int(match[0])
        val_loss = float(match[1])
        val_ppl = float(match[2])
        val_te_mse = float(match[3])
        
        val_epochs.append(epoch)
        val_losses.append(val_loss)
        val_ppls.append(val_ppl)
        val_te_mses.append(val_te_mse)
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'train_lm_losses': train_lm_losses,
        'train_reg_losses': train_reg_losses,
        'val_epochs': val_epochs,
        'val_losses': val_losses,
        'val_ppls': val_ppls,
        'val_te_mses': val_te_mses
    }

def plot_losses(log_data, output_dir):
    """Plot loss curves"""
    sns.set(style="whitegrid")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 1. Overall loss curves
    plt.figure(figsize=(12, 6))
    plt.plot(log_data['epochs'], log_data['train_losses'], label='Training Loss', marker='o', markersize=3)
    plt.plot(log_data['val_epochs'], log_data['val_losses'], label='Validation Loss', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / 'total_loss.png', dpi=300)
    plt.close()
    
    # 2. Training loss components
    plt.figure(figsize=(12, 6))
    plt.plot(log_data['epochs'], log_data['train_losses'], label='Total Training Loss', marker='o', markersize=3)
    plt.plot(log_data['epochs'], log_data['train_lm_losses'], label='Language Model Loss', marker='s', markersize=3)
    plt.plot(log_data['epochs'], log_data['train_reg_losses'], label='Regression Loss', marker='^', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Components vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / 'train_loss_components.png', dpi=300)
    plt.close()
    
    # 3. Validation loss components
    plt.figure(figsize=(12, 6))
    plt.plot(log_data['val_epochs'], log_data['val_losses'], label='Total Validation Loss', marker='o', markersize=3)
    plt.plot(log_data['val_epochs'], log_data['val_te_mses'], label='TE MSE', marker='^', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss and TE MSE vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / 'val_loss_components.png', dpi=300)
    plt.close()
    
    # 4. Perplexity changes
    plt.figure(figsize=(12, 6))
    plt.plot(log_data['val_epochs'], log_data['val_ppls'], label='Validation Perplexity', marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity vs Epoch')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / 'val_perplexity.png', dpi=300)
    plt.close()
    
    # 5. Save data to CSV file
    train_df = pd.DataFrame({
        'epoch': log_data['epochs'],
        'train_loss': log_data['train_losses'],
        'train_lm_loss': log_data['train_lm_losses'],
        'train_reg_loss': log_data['train_reg_losses']
    })
    
    val_df = pd.DataFrame({
        'epoch': log_data['val_epochs'],
        'val_loss': log_data['val_losses'],
        'val_ppl': log_data['val_ppls'],
        'val_te_mse': log_data['val_te_mses']
    })
    
    # Merge dataframes
    merged_df = pd.merge(train_df, val_df, on='epoch', how='outer')
    merged_df.to_csv(output_path / 'training_metrics.csv', index=False)
    
    print(f"Charts saved to {output_path} directory")

def main():
    parser = argparse.ArgumentParser(description="Visualize training and validation losses")
    parser.add_argument("--log_file", type=str, required=True, help="Path to training log file")
    parser.add_argument("--output_dir", type=str, default="./loss_plots", help="Directory for output charts")
    args = parser.parse_args()
    
    # Parse log file
    log_data = parse_log_file(args.log_file)
    
    # Plot loss curves
    plot_losses(log_data, args.output_dir)

if __name__ == "__main__":
    main() 