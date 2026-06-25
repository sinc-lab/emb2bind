import os
import re 
import torch as tr
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
from pathlib import Path


def read_and_clean_fasta(fasta_path: str) -> tuple[list[str], list[str]]:
    """
    Read a FASTA file and clean protein sequences by replacing rare amino acids.
    """
    with open(fasta_path, "r", encoding="utf-8") as handle:
        records = list(SeqIO.parse(handle, "fasta"))
    
    # Clean sequences (replace rare amino acids with X) and extract their IDs
    sequences = ["".join(list(re.sub(r"[BZJUO]", "X", str(record.seq)))) for record in records]
    protein_ids = [record.id for record in records]
    
    return protein_ids, sequences


def get_esm2(
        fasta_path: str, 
        output_dir: str, 
        model_name: str = "esm2_t6_8M_UR50D", 
        device: str = 'cuda',
        max_length: int = 4000
        ):
    """
    Generate ESM2 embeddings for protein sequences in a FASTA file and save them as .npy files.
    """
    import esm
    
    # Available ESM2 models and their representation layers
    MODELS = {
        "esm2_t33_650M_UR50D": 33,
        "esm2_t6_8M_UR50D": 6
    }
    
    device = tr.device(device)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model directly
    if model_name == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    elif model_name == "esm2_t6_8M_UR50D":
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    num_repr_layer = MODELS[model_name]  

    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Read all protein sequences from the FASTA file
    protein_ids, sequences = read_and_clean_fasta(fasta_path)
    
    skipped_proteins = []
    for prot_id, seq in tqdm(zip(protein_ids, sequences)):
        # Check sequence length
        if len(seq) > max_length:
            skipped_proteins.append((prot_id, len(seq)))
            continue
        
        data = [(prot_id, seq)]          

        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)
        
        with tr.no_grad():
            results = model(batch_tokens, repr_layers=[num_repr_layer], return_contacts=False)
        
        for i, embed in enumerate(results["representations"][num_repr_layer]):
            new_embed = embed.cpu().numpy()[1:batch_lens[0]-1]
            new_embed = new_embed.T
            np.save(os.path.join(output_dir, f'{prot_id}.npy'), arr=new_embed)
    
    # Print summary of skipped proteins
    if skipped_proteins:
        print(f"\nSkipped {len(skipped_proteins)} proteins with length > {max_length} aa:")
        for prot_id, length in skipped_proteins:
            print(f"  - {prot_id}: {length} aa")


def load_plm_embedding(acc: str, plm_emb_dir: str):
    """Load the precomputed pLM embedding for a protein accession."""
    plm_path = Path(plm_emb_dir) / f"{acc}.npy"
    
    if not plm_path.exists():
        raise FileNotFoundError(f"pLM embedding not found: {plm_path}")
    return np.load(plm_path)