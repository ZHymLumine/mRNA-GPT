import os
import csv
import re

input_dir = "/home/yzhang/research/mRNAdesigner_3/data"
output_file = "/home/yzhang/research/mRNAdesigner_3/data/rna_seq_test.txt"

with open(output_file, "w") as out_f:
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".fna"):
                fna_path = os.path.join(root, file)
                
                if os.path.exists(fna_path):
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
