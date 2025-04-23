import os
import csv

input_dir = "/Users/zym/Downloads/Research/Okumura_lab/protein2rna/ncbi_dataset/data"
output_file = "/Users/zym/Downloads/Research/Okumura_lab/mRNAdesigner_2/data/rna_seq.txt"
data_summary_file = "/Users/zym/Downloads/Research/Okumura_lab/mRNAdesigner_2/data/filtered_bacteria_species_updated_final.csv"

taxonomy_info = {}
with open(data_summary_file, "r") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        gcf_id = row["Accession"] or "unk"
        tax_id = row.get("Tax_ID", "unk") or "unk"
        superkingdom = row.get("superkingdom", "unk") or "unk"
        kingdom = row.get("kingdom", "unk") or "unk"
        phylum = row.get("phylum", "unk") or "unk"
        class_ = row.get("class", "unk") or "unk"
        order = row.get("order", "unk") or "unk"
        family = row.get("family", "unk") or "unk"
        genus = row.get("genus", "unk") or "unk"
        species = row.get("species", "unk") or "unk"
        strain = row.get("Strain", "unk") or "unk"

        print(f'strain in summary: {strain}')
        taxonomy_info[gcf_id] = {
            "tax_id": tax_id,
            "superkingdom": superkingdom,
            "kingdom": kingdom,
            "phylum": phylum,
            "class": class_,
            "order": order,
            "family": family,
            "genus": genus,
            "species": species,
            "strain": strain,
        }

with open(output_file, "w") as out_f:
    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            if dir_name.startswith("GCF_"):
                fna_path = os.path.join(root, dir_name, "cds_from_genomic.fna")
                
                if os.path.exists(fna_path):
                    gcf_id = dir_name
                    tax_info = taxonomy_info.get(gcf_id, {})
                    
                    tax_id = tax_info.get("tax_id", "unk")
                    superkingdom = tax_info.get("superkingdom", "unk")
                    kingdom = tax_info.get("kingdom", "unk")
                    phylum = tax_info.get("phylum", "unk")
                    class_ = tax_info.get("class", "unk")
                    order = tax_info.get("order", "unk")
                    family = tax_info.get("family", "unk")
                    genus = tax_info.get("genus", "unk")
                    species = tax_info.get("species", "unk")
                    strain = tax_info.get("strain", "unk")

                    print(f'strain: {strain}')
                    with open(fna_path, "r") as fna_file:
                        sequence = ""
                        for line in fna_file:
                            if line.startswith(">"):
                                if sequence:
                                    rna_sequence = sequence.replace("T", "U")
                                    out_f.write(f"{rna_sequence},{tax_id},{superkingdom},{kingdom},{phylum},{class_},{order},{family},{genus},{species},{strain}\n")
                                    sequence = ""
                            else:
                                sequence += line.strip()
                        if sequence:
                            rna_sequence = sequence.replace("T", "U")
                            out_f.write(f"{rna_sequence},{tax_id},{superkingdom},{kingdom},{phylum},{class_},{order},{family},{genus},{species},{strain}\n")
    print(f"Done processing and save to {output_file}")
