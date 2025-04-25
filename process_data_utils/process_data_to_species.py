import os
import csv
import re

input_dir = "/Users/zym/Downloads/Research/Okumura_lab/protein2rna/ncbi_dataset/data"
output_file = "/Users/zym/Downloads/Research/Okumura_lab/mRNAdesigner_3/data/rna_seq.txt"

with open(output_file, "w") as out_f:
    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            if dir_name.startswith("GCF_"):
                fna_path = os.path.join(root, dir_name, "cds_from_genomic.fna")
                
                if os.path.exists(fna_path):
                    gcf_id = dir_name
                    
                    with open(fna_path, "r") as fna_file:
                        sequence = ""
                        for line in fna_file:
                            if line.startswith(">"):
                                if sequence:
                                    # Check if sequence contains only A, U, C, G and is a multiple of 3
                                    if re.fullmatch(r"[AUCG]+", sequence) and len(sequence) % 3 == 0:
                                        # Organize into codons
                                        codon_sequence = ' '.join(sequence[i:i+3] for i in range(0, len(sequence), 3))
                                        out_f.write(f"{codon_sequence}\n")
                                    sequence = ""
                            else:
                                sequence += line.strip().replace("T", "U")
                        if sequence:
                            if re.fullmatch(r"[AUCG]+", sequence) and len(sequence) % 3 == 0:
                                codon_sequence = ' '.join(sequence[i:i+3] for i in range(0, len(sequence), 3))
                                out_f.write(f"{codon_sequence}\n")
    print(f"Done processing and save to {output_file}")
