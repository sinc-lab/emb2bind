"""
Predict binding residues from protein sequences using a trained ensemble model and
protein language models.

Usage:
    python predict.py -f data/samples.fasta
"""
import os
import yaml
import time
import argparse
import numpy as np
import torch as tr
from pathlib import Path

from src.ensemble import EnsembleModel, build_ensemble_dirs
from src.energy import AIUPredTransformer, calculate_energy_embedding, load_energy_model
from src.plms import load_plm_embedding
from src.utils import read_fasta
from src.caid_output import format_caid_rows, save_protein_prediction_caid, save_combined_predictions, save_prediction_timings


def parser():
    parser = argparse.ArgumentParser(
        description='Predict binding residues from protein sequences',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--fasta', '-f',
        type=str,
        required=True,
        help='Path to FASTA file containing protein sequences to predict.'
    )
    parser.add_argument(
        '--embedding-dir', '-e',
        type=str,
        default='data/embeddings/',
        help='Directory containing precomputed pLM embeddings.'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results/binding/',
        help='Output directory to save predictions.'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/predictions.yaml',
        help='Path to YAML configuration file for predictions.'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cpu',
        help='Device to run predictions on (e.g., "cpu", "cuda", "cuda:0", "cuda:1")'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=4,
        help='Cap the number of CPU threads torch uses (torch.set_num_threads).'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    return parser.parse_args()


def set_threads(num_threads: int = 4):
    """Set the number of threads for torch and other libraries."""
    n_str = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = n_str
    os.environ["MKL_NUM_THREADS"] = n_str
    os.environ["OPENBLAS_NUM_THREADS"] = n_str
    os.environ["NUMEXPR_NUM_THREADS"] = n_str

    tr.set_num_threads(num_threads) # threads for intra-operation parallelism


def build_embedding(
        acc: str,
        sequence: str,
        plm_emb_dir: str = "data/embeddings/",
        energy_model: AIUPredTransformer = None,
        device: str = 'cpu',
        ):
    """
    Build a combined embedding using the pLM embedding and energy embeddings,
    generated on-the-fly.
    """
    # The scale factor was determined empirically to bring the energy embedding 
    # values into a similar range as the pLM embeddings.
    SCALE_FACTOR = 36.8080

    plm_emb = load_plm_embedding(acc, plm_emb_dir)       # (emb_dim_plm, L)

    energy_tensor = calculate_energy_embedding(sequence, energy_model, 
                                               device, scale_factor=SCALE_FACTOR)
    energy_emb = energy_tensor.detach().cpu().numpy()    # (emb_dim_energy, L)

    emb = np.concatenate([energy_emb, plm_emb], axis=0)  # (emb_dim_energy + emb_dim_plm, L)
    return tr.tensor(emb, dtype=tr.float32)


def run_for_protein(
        acc: str,
        sequence: str,
        model: EnsembleModel,
        plm_emb_dir: str,
        energy_model: AIUPredTransformer,
        device: str,
        window_step: int = 1,
        threshold: float = 0.5,
        verbose: bool = False,
        ):
    """Run prediction pipeline for a single protein accession."""
    start_time = time.time()

    try:
        emb = build_embedding(acc, sequence, plm_emb_dir=plm_emb_dir,
            energy_model=energy_model, device=device)
    except Exception as e:
        print(f"Skipping {acc}: could not build embeddings - {e}")
        return None

    if verbose:
        print(f"\nProtein ID: {acc}")
        print(f"\tEmbedding shape: {emb.shape}")
        print(f"\tSequence length: {emb.shape[1]} residues")

    centers, predictions = model.pred_sliding_window(emb, step=window_step)

    elapsed_ms = int((time.time() - start_time) * 1000)
    scores = predictions[:, 1]
    labels = (scores > threshold).astype(int)
    return {
        'protein_id': acc,
        'rows': format_caid_rows(centers, sequence, scores, labels),
        'elapsed_ms': elapsed_ms,
    }


def main():
    args = parser()

    set_threads(args.threads)

    verbose = args.verbose
    device = args.device
    plm_emb_dir = args.embedding_dir
    output_dir = Path(args.output_dir)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Prediction config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    # Process multiple proteins from the input FASTA
    energy_model_weights = config.get('energy_model_weights', 'models/AIUPred/embedding.pt')
    energy_model = load_energy_model(energy_model_weights, device)

    # Use an ensemble of trained models (list of model folders)
    ensemble_dirs = build_ensemble_dirs(config)
    if verbose:
        print("Ensemble model directories:")
        for ensemble_model_dir in ensemble_dirs:
            print(f"  - {ensemble_model_dir}")
    model = EnsembleModel(ensemble_dirs, device=device)

    fasta_records = read_fasta(args.fasta)
    if not fasta_records:
        raise ValueError(f"No sequences found in FASTA: {args.fasta}")
    threshold = config.get('threshold', 0.5)

    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    timings = []
    print(f"Predicting binding residues for {len(fasta_records)} proteins...")
    for header, sequence in fasta_records.items():
        acc_from_header = header.split()[0]

        result = run_for_protein(
            acc_from_header,
            sequence,
            model,
            plm_emb_dir,
            energy_model,
            device,
            threshold=threshold,
            verbose=verbose
        )

        if result is None:
            timings.append((acc_from_header, 0))
            continue
        
        save_protein_prediction_caid(result['protein_id'], result['rows'], output_dir)
        all_rows.append(result)
        timings.append((acc_from_header, result['elapsed_ms']))

    print(f"\nAll individual outputs saved to: {output_dir}")

    # Write a combined CAID file with all predictions
    combined_caid = save_combined_predictions(all_rows, output_dir)
    print(f"Combined predictions saved to: {combined_caid}")

    timings_csv = save_prediction_timings(timings, output_dir)
    if timings_csv is not None:
        print(f"Timings written to: {timings_csv}")
    print()


if __name__ == '__main__':
    main()