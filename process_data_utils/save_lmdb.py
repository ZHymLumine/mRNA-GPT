import os
import numpy as np
from transformers import AutoTokenizer, BertTokenizerFast
import random
import argparse
import psutil
import lmdb
from tqdm import tqdm
import pickle
import sys

def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    print(f"Current memory usage: {memory:.2f} MB")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process the text data for tokenization.')
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the trained AutoTokenizer.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory of output files.")
    parser.add_argument("--block_size", type=int, default=1024, help="Max token length.")
    parser.add_argument("--is_start_with_eos", type=bool, default=True, help="Whether each line starts with `eos_token`.")
    parser.add_argument("--is_end_with_eos", type=bool, default=True, help="Whether each line ends with `eos_token`.")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train-validation split ratio.")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Number of lines to process per chunk.")
    return parser.parse_args()

def save_to_lmdb(env, data, start_idx):
    """
    Save tokenized data to an LMDB environment.
    Args:
        env: LMDB environment.
        data: List of tokenized sequences to save.
        start_idx: Starting index for keys in LMDB.
    """
    with env.begin(write=True) as txn:
        for idx, ids in enumerate(data):
            txn.put(f"{start_idx + idx}".encode('utf-8'), np.array(ids, dtype=np.int32).tobytes())

def save_to_lmdb(env, data, start_idx):
    """
    Save tokenized data to an LMDB environment, dynamically adjusting map_size if needed.
    """
    try:
        with env.begin(write=True) as txn:
            for idx, ids in enumerate(data):
                key = f"{start_idx + idx}".encode('utf-8')
                value = np.array(ids, dtype=np.int32).tobytes()
                txn.put(key, value)
    except lmdb.MapFullError:
        current_size = env.info()['map_size']
        new_size = current_size * 2  # 将 map_size 增加一倍
        env.set_mapsize(new_size)
        print(f"Map size increased to {new_size / (1024**3):.2f} GB.")
        save_to_lmdb(env, data, start_idx)

def read_lmdb(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )  
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    out_list = []
    for idx in tqdm(keys):
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
        out_list.append(data)
    env.close()
    return out_list

def read_lmdb_numpy(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )  
    out_list = []
    with env.begin() as txn:
        cursor = txn.cursor() 
        for key, value in cursor:
            data = np.frombuffer(value, dtype=np.int32)
            out_list.append(data)
    
    env.close()
    return out_list

def count_lmdb_rows(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )

    stat = env.stat()  # Get database statistics
    print(f"stat:{stat}")
    row_count = stat['entries']  # Total number of entries
    env.close()
    return row_count

# input format: [SEP]<mRNA sequence>[SEP]
def tokenize_and_save_lines(tokenizer, input_file, train_txt_file, val_txt_file, train_lmdb_path, val_lmdb_path, is_start_with_eos, is_end_with_eos, block_size, split_ratio, chunk_size):
    def process_line(line):
        parts = line.strip().split(',')
        text = parts[0]
        
        if is_start_with_eos:
            text = "[SEP]" + text
        if is_end_with_eos:
            text = text + "[SEP]"
        return text

    train_lines = []
    val_lines = []
    train_ids = []
    val_ids = []

    train_env = lmdb.open(train_lmdb_path, subdir=False, readonly=False, lock=False, readahead=False, meminit=False, map_size=10**10)
    val_env = lmdb.open(val_lmdb_path, subdir=False, readonly=False, lock=False, readahead=False, meminit=False, map_size=10**10)

    train_idx = 0
    val_idx = 0

    # Initialize text files
    for path in [train_txt_file, val_txt_file]:
        open(path, "w").close()

    # Process file in chunks
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(line)
            if len(lines) == chunk_size or not line:
                random.shuffle(lines)
                split_at = int(split_ratio * len(lines))
                train_lines_list = lines[:split_at]
                val_lines_list = lines[split_at:]

                for train_line in train_lines_list:
                    text = process_line(train_line)
                    if text != '':
                        ids = tokenizer.encode(text)
                        if len(ids) <= block_size:
                            train_ids.append(ids)
                            train_lines.append(text)

                for val_line in val_lines_list:
                    text = process_line(val_line)
                    if text != '':
                        ids = tokenizer.encode(text)
                        if len(ids) <= block_size:
                            val_ids.append(ids)
                            val_lines.append(text)

                # print_memory_usage()
                # Save each chunk to LMDB
                save_to_lmdb(train_env, train_ids, train_idx)
                save_to_lmdb(val_env, val_ids, val_idx)
                train_idx += len(train_ids)
                val_idx += len(val_ids)
                
                # print_memory_usage()
                # Save raw text lines to txt files
                with open(train_txt_file, "a", encoding="utf-8") as train_txt:
                    train_txt.write("\n".join(train_lines) + "\n")
                with open(val_txt_file, "a", encoding="utf-8") as val_txt:
                    val_txt.write("\n".join(val_lines) + "\n")

                # print_memory_usage()
                lines = []  # Clear chunk buffer
                train_lines.clear()
                val_lines.clear()
                train_ids.clear()
                val_ids.clear()
                print(f"Processed {i + 1} lines...")

    train_env.close()
    val_env.close()
    print("Tokenization and data saving completed.")

def main():
    args = parse_arguments()
    set_random_seed(42)

    # Paths setup
    raw_data_path = args.file_path
    train_txt_path = os.path.join(args.out_dir, 'train_codon.txt')
    val_txt_path = os.path.join(args.out_dir, 'val_codon.txt')
    train_lmdb_path = os.path.join(args.out_dir, 'train_codon.lmdb')
    val_lmdb_path = os.path.join(args.out_dir, 'val_codon.lmdb')
    print("Paths setup complete...")

    # Tokenization
    VOCAB_FILE = os.path.join(args.tokenizer_path, "vocab.txt")
    tokenizer = BertTokenizerFast(vocab_file=VOCAB_FILE, do_lower_case=False)
    tokenize_and_save_lines(tokenizer, raw_data_path, train_txt_path, val_txt_path, train_lmdb_path, val_lmdb_path, args.is_start_with_eos, args.is_end_with_eos, args.block_size, args.split_ratio, args.chunk_size)

    num_rows = count_lmdb_rows(train_lmdb_path)
    print(f"Number of rows in the training LMDB file: {num_rows}")
    num_rows = count_lmdb_rows(val_lmdb_path)
    print(f"Number of rows in the validation LMDB file: {num_rows}")

if __name__ == "__main__":
    main()

    

# python save_lmdb.py --file_path /Users/zym/Downloads/Research/Okumura_lab/mRNAdesigner_3/data/rna_seq.txt --tokenizer_path "/Users/zym/Downloads/Research/Okumura_lab/mRNAdesigner_3/tokenizer" --out_dir /Users/zym/Downloads/Research/Okumura_lab/mRNAdesigner_3/data/ --block_size 512