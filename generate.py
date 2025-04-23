import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import argparse


def main(args):
    compile = False
    # load model and tokenizer'
    tokenizer = AutoTokenizer.from_pretrained("ZYMScott/mRNAdesigner_BPE_1024")
    # tokenizer.add_special_tokens({"pad_token": "<|pad|>"}) 

    model_path = "/raid_elmo/home/lr/zym/mRNAdesigner/checkpoints/gpt2_test/"
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    # model.resize_token_embeddings(len(tokenizer))
    model.eval()

    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    taxonomy_text = "".join([f"<{value}>" for value in [
        args.tax_id, args.kingdom, args.phylum, args.class_, args.order, args.family, args.genus, args.species, args.strain
    ]])

    input_text = f"{taxonomy_text}<|endoftext|>" 
    # input_text = "<|endoftext|>" 
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    output = model.generate(
        input_ids,
        max_length=args.max_length,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=5,
        do_sample=True,
        top_k=50, 
        top_p=0.95,
        temperature=0.7,
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated RNA sequence:", generated_text)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Generate RNA sequence based on taxonomy information")
    parser.add_argument("--tax_id", type=str, required=False, help="Taxonomy: Tax id")
    parser.add_argument("--kingdom", type=str, required=True, help="Taxonomy: Kingdom")
    parser.add_argument("--phylum", type=str, required=True, help="Taxonomy: Phylum")
    parser.add_argument("--class_", type=str, required=True, help="Taxonomy: Class")  # 注意：避免与 Python 的关键字 `class` 冲突
    parser.add_argument("--order", type=str, required=True, help="Taxonomy: Order")
    parser.add_argument("--family", type=str, required=True, help="Taxonomy: Family")
    parser.add_argument("--genus", type=str, required=True, help="Taxonomy: Genus")
    parser.add_argument("--species", type=str, required=True, help="Taxonomy: Species")
    parser.add_argument("--strain", type=str, required=False, help="Taxonomy: Strain")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated sequence")
    args = parser.parse_args()

    main(args)
    

# CUDA_VISIBLE_DEVICES=0 python generate.py --tax_id "511145" --kingdom "Bacteria" --phylum "Pseudomonadota" --class_ "Gammaproteobacteria" --order "Enterobacterales" --family "Enterobacteriaceae" --genus "Escherichia" --species "Escherichia coli" --strain "K-12 substr. MG1655" --max_length 1000