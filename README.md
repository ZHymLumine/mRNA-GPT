# mRNAdesigner

A novel foundational pre-trained generative language model for efficient mRNA design

<details><summary>Table of contents</summary>
  
- [Setup Environment](#Setup_Environment)
- [Pre-trained Models](#Available_Pretrained_Models)
- [Usage](#usage)
  - [Train Tokenizer](#tokenizer)
  - [Pretrain](#pretrain)
- [License](#license)
</details>

## Create Environment with Conda <a name="Setup_Environment"></a>

First, download the repository and create the environment.

```
git clone https://github.com/ZHymLumine/mRNAdesigner.git
cd ./mRNAdesigner
conda create -n mRNAdesigner python=3.10 -y
conda activate mRNAdesigner
bash environment.sh
```

install flash attention 2 (optional)

`pip install -U flash-attn --no-build-isolation`

## Apply mRNAdesigner with Existing Scripts. <a name="Usage"></a>

### 0. Data preparation

```
  python process_data_to_species.py
```

### 1. train your own tokenizer <a name="tokenizer"></a>

you can train your own tokenizer following the script

```
python codon_tokenizer.py \
  --txt_file_path /raid_elmo/home/lr/zym/mRNAdesigner/data/rna_seq.txt  \
  --new_tokenizer_path ./mRNAdeisgner_codon \
  --species_file_path /home/lr/zym/research/mRNAdesigner/data/species.csv
```

or use our tokenizer in code

```
tokenizer = AutoTokenizer.from_pretrained("ZYMScott/mRNAdesigner")
```

#### training

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --config ./configs/pretrain.yaml
```

check ip address

`hostname -I | awk '{print $1}'`
