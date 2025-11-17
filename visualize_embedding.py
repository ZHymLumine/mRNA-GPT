import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from transformers import BertTokenizerFast
from model import GPTConfig, GPT
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm
import matplotlib.font_manager as fm

# Font configuration
# Liberation Serif is metrically compatible with Times New Roman
# It has identical character widths and spacing, making it the perfect substitute
# for academic publications. Widely accepted by journals and publishers.
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'Times New Roman', 'DejaVu Serif', 'Times']
plt.rcParams['mathtext.fontset'] = 'stix'  # STIX fonts are similar to Times
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Force matplotlib to rebuild font cache
import matplotlib
matplotlib.font_manager._load_fontmanager(try_read_cache=False)

print(f"Using font: {plt.rcParams['font.serif'][0]}")

def load_pretrained_model(checkpoint_path):
    """Load the pretrained model"""
    print("Loading pretrained model...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Remove _orig_mod prefix from state dict
    def remove_orig_mod_prefix(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[len('_orig_mod.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict
    
    clean_state_dict = remove_orig_mod_prefix(checkpoint['model'])
    
    # Create model
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(clean_state_dict)
    model.eval()
    
    print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model

def read_fasta_sequences(file_path, num_sequences=500):
    """Read sequences from FASTA file with random sampling"""
    # First, read all sequences
    all_sequences = []
    with open(file_path, 'r') as handle:
        for record in SeqIO.parse(handle, "fasta"):
            all_sequences.append(str(record.seq))
    
    # Randomly sample num_sequences if there are more sequences than needed
    if len(all_sequences) > num_sequences:
        # sequences = random.sample(all_sequences, num_sequences)
        sequences = all_sequences[:num_sequences]
    else:
        sequences = all_sequences
    
    return sequences

def dna_to_rna(dna_sequence):
    """Convert DNA sequence to RNA (T -> U), handles both uppercase and lowercase"""
    # Convert to uppercase first, then replace T with U
    return dna_sequence.upper().replace('T', 'U')

def split_sequence_to_codons(sequence):
    """Split RNA sequence into codons (triplets) and format like training data"""
    # Remove any non-ACGU characters and convert to uppercase
    clean_seq = ''.join(c for c in sequence.upper() if c in 'ACGU')
    
    # Split into codons
    codons = []
    for i in range(0, len(clean_seq) - 2, 3):
        codon = clean_seq[i:i+3]
        if len(codon) == 3:  # Only add complete codons
            codons.append(codon)
    
    # Format like training data with [SEP] tokens
    if codons:
        return '[SEP]' + ' '.join(codons) + '[SEP]'
    else:
        return ''

def get_sequence_embedding(model, tokenizer, sequence, device='cpu', max_length=1024):
    """Get embedding for a single sequence using last hidden states"""
    try:
        # Tokenize the sequence
        inputs = tokenizer(
            sequence,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(device)
        
        with torch.no_grad():
            # Forward pass through the model to get hidden states
            # We need to modify the forward pass to get intermediate outputs
            x = model.transformer.wte(input_ids)  # Token embeddings
            pos = torch.arange(0, input_ids.size(1), dtype=torch.long, device=device)
            pos_emb = model.transformer.wpe(pos)  # Position embeddings
            x = model.transformer.drop(x + pos_emb)
            
            # Pass through all transformer blocks
            for block in model.transformer.h:
                x = block(x)
            
            # Apply final layer norm to get the last hidden states
            last_hidden_states = model.transformer.ln_f(x)
            
            # Use mean pooling over sequence length to get fixed-size embedding
            # Shape: (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
            # Only pool over non-padding tokens
            attention_mask = (input_ids != tokenizer.pad_token_id).float()
            masked_hidden_states = last_hidden_states * attention_mask.unsqueeze(-1)
            embedding = masked_hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
        return embedding.cpu().numpy().flatten()
    
    except Exception as e:
        print(f"Error processing sequence: {e}")
        return None

def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Configuration for all three domains
    domains_config = {
        'archaea': {
            'data_dir': "/home/yzhang/research/mRNAdesigner_3/data/archaea",
            'checkpoint_path': "/home/yzhang/research/mRNAdesigner_3/checkpoints/pretrained_archaea_ckpt_62000.pt",
            'species_files': {
                'Aeropyrum camini': 'Aeropyrum_camini_cds_from_genomic.fna',
                'Haloarcula marina': 'Haloarcula_marina_cds_from_genomic.fna',
                'Halomarina oriensis': 'Halomarina_oriensis_cds_from_genomic.fna',
                'Halorussus halobius': 'Halorussus_halobius_cds_from_genomic.fna',
                'Methanospirillum lacunae': 'Methanospirillum_lacunae_cds_from_genomic.fna',
                'Thermococcus gammatolerans': 'Thermococcus_gammatolerans_cds_from_genomic.fna'
            }
        },
        'bacteria': {
            'data_dir': "/home/yzhang/research/mRNAdesigner_3/data/bacteria",
            'checkpoint_path': "/home/yzhang/research/mRNAdesigner_3/checkpoints/pretrained_bacteria_ckpt_563000.pt",
            'species_files': {
                'Bacillus subtilis': 'Bacillus_subtilis_cds_from_genomic.fna',
                'Escherichia coli': 'Escherichia_coli_cds_from_genomic.fna',
                'Helicobacter pylori': 'Helocobacter_pylori_cds_from_genomic.fna',
                'Mycobacterium tuberculosis': 'Mycobacterium_tuberculosis_cds_from_genomic.fna',
                'Pseudomonas aeruginosa': 'Pseudomonas_aeruginosa_cds_from_genomic.fna',
                'Staphylococcus aureus': 'Staphylococcus_aureus_cds_from_genomic.fna'
            }
        },
        'eukaryote': {
            'data_dir': "/home/yzhang/research/mRNAdesigner_3/data/eukaryote",
            'checkpoint_path': "/home/yzhang/research/mRNAdesigner_3/checkpoints/pretrained_eukaryote_ckpt_624000.pt",
            #  'checkpoint_path': '/home/yzhang/research/mRNAdesigner_3/checkpoints/ckpt_563000.pt',
            'species_files': {
                'Arabidopsis thaliana': 'Arabidopsis_thaliana_cds.fna',
                'Candida albicans': 'Candida_albicans_cds_from_genomic.fna',
                'Homo sapiens': 'Homo_sapiens_cds_from_genomic.fna',
                'Mus musculus': 'Mus_musculus_cds_from_genomic.fna',
                'Saccharomyces cerevisiae': 'Saccharomyces cerevisiae_cds_from_genomic.fna',
                'Zea mays': 'Zea_mays_cds_from_genomic.fna'
            }
        }
    }
    
    vocab_file = "/home/yzhang/research/mRNAdesigner_3/tokenizer/vocab.txt"
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BertTokenizerFast(vocab_file=vocab_file, do_lower_case=False)
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    # Collect all sequences and labels
    all_embeddings = []
    all_labels = []
    all_domains = []
    
    print("\nProcessing sequences from all three domains...")
    
    # Process each domain
    for domain_name, domain_config in domains_config.items():
        print(f"\n{'='*60}")
        print(f"Processing domain: {domain_name.upper()}")
        print(f"{'='*60}")
        
        data_dir = domain_config['data_dir']
        checkpoint_path = domain_config['checkpoint_path']
        species_files = domain_config['species_files']
        
        # Load model for this domain
        print(f"\nLoading {domain_name} model...")
        model = load_pretrained_model(checkpoint_path)
        model.to(device)
        
        # Process each species in this domain
        for species_name, filename in species_files.items():
            file_path = os.path.join(data_dir, filename)
            print(f"\nProcessing {species_name}...")
            
            # Read DNA sequences (500 per species)
            dna_sequences = read_fasta_sequences(file_path, num_sequences=500)
            print(f"Read {len(dna_sequences)} DNA sequences")
            
            # Convert to RNA and process
            species_embeddings = []
            valid_count = 0
            
            for i, dna_seq in enumerate(tqdm(dna_sequences, desc=f"Processing {species_name}")):
                # Convert DNA to RNA
                rna_seq = dna_to_rna(dna_seq)
                
                # Split into codons
                codon_seq = split_sequence_to_codons(rna_seq)
                
                if len(codon_seq.strip()) == 0:  # Skip empty sequences
                    continue
                    
                # Get embedding
                embedding = get_sequence_embedding(model, tokenizer, codon_seq, device)
                
                if embedding is not None:
                    species_embeddings.append(embedding)
                    valid_count += 1
                    
            
            print(f"Generated {len(species_embeddings)} embeddings for {species_name}")
            
            # Add to overall collections with domain label
            all_embeddings.extend(species_embeddings)
            all_labels.extend([f"{domain_name}_{species_name}"] * len(species_embeddings))
            all_domains.extend([domain_name] * len(species_embeddings))
        
        # Free up GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\nTotal embeddings collected: {len(all_embeddings)}")
    
    # Convert to numpy array
    embeddings_array = np.array(all_embeddings)
    print(f"Embeddings shape: {embeddings_array.shape}")
    
    # Standardize embeddings
    print("Standardizing embeddings...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_array)
    
    # Apply PCA first to reduce dimensionality
    print("Applying PCA dimensionality reduction...")
    pca_components = 50
    pca = PCA(n_components=pca_components, random_state=42)  # Reduce to 50 dimensions first
    embeddings_pca = pca.fit_transform(embeddings_scaled)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Apply UMAP on PCA-reduced embeddings
    print("Applying UMAP dimensionality reduction...")
    umap_reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=42
    )
    
    # USE SCALING
    # embeddings_2d = umap_reducer.fit_transform(embeddings_scaled)
    # USE PCA
    embeddings_2d = umap_reducer.fit_transform(embeddings_pca)
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'UMAP1': embeddings_2d[:, 0],
        'UMAP2': embeddings_2d[:, 1],
        'Species': all_labels,
        'Domain': all_domains
    })
    
    # Define consistent colors for all domains (6 highly distinguishable colors)
    # Based on colorblind-friendly palette with maximum contrast
    # Red, Blue, Green, Orange, Purple/Magenta, Black
    # consistent_colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#000000']
    
    consistent_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    # Create separate visualization for each domain
    for domain_name, domain_config in domains_config.items():
        plt.figure(figsize=(12, 8))
        
        species_list = list(domain_config['species_files'].keys())
        
        # Plot only this domain's species
        for i, species in enumerate(species_list):
            label = f"{domain_name}_{species}"
            species_data = df[df['Species'] == label]
            plt.scatter(
                species_data['UMAP1'], 
                species_data['UMAP2'],
                c=consistent_colors[i % len(consistent_colors)],
                label=species,  # Use original species name with spaces
                alpha=0.6,
                s=30,
                edgecolors='none'
            )
        
        plt.title(f'UMAP Visualization of {domain_name.capitalize()} RNA Sequence Embeddings\n(Last Hidden States + PCA + UMAP)')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        
        # Create legend with italic species names
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, frameon=True, fancybox=False)
        # Make legend text italic
        for text in legend.get_texts():
            text.set_style('italic')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot for this domain in multiple formats
        base_path = f'/home/yzhang/research/mRNAdesigner_3/species_embeddings/{domain_name}_embeddings_pca{pca_components}'
        
        # Save as high-resolution PNG (raster)
        plt.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight')
        print(f"{domain_name.capitalize()} PNG saved to: {base_path}.png")
        
        # Save as PDF (vector - recommended for papers)
        plt.savefig(f'{base_path}.pdf', format='pdf', bbox_inches='tight')
        print(f"{domain_name.capitalize()} PDF saved to: {base_path}.pdf")
        
        # Optional: Save as SVG (vector - for web/presentations)
        plt.savefig(f'{base_path}.svg', format='svg', bbox_inches='tight')
        print(f"{domain_name.capitalize()} SVG saved to: {base_path}.svg")
        
        plt.close()
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total sequences processed: {len(all_embeddings)}")
    print("\nSequences per domain:")
    for domain in ['archaea', 'bacteria', 'eukaryote']:
        count = sum(1 for d in all_domains if d == domain)
        print(f"  {domain}: {count}")
    
    print("\nSequences per species:")
    for domain_name, domain_config in domains_config.items():
        print(f"\n  {domain_name.upper()}:")
        for species in domain_config['species_files'].keys():
            label = f"{domain_name}_{species}"
            count = sum(1 for l in all_labels if l == label)
            print(f"    {species}: {count}")
    
    # Save embeddings and metadata
    results = {
        'embeddings_2d': embeddings_2d,
        'embeddings_pca': embeddings_pca,
        'embeddings_original': embeddings_array,
        'labels': all_labels,
        'domains': all_domains,
        'domains_config': str(domains_config),
        'pca_explained_variance_ratio': pca.explained_variance_ratio_
    }
    
    results_path = '/home/yzhang/research/mRNAdesigner_3/species_embeddings/three_domains_embeddings_results.npz'
    np.savez(results_path, **results)
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
