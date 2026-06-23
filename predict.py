"""
Predict disorder from protein embeddings using a trained ensemble model.

Run:
    python predict_CAID.py -f data/samples.fasta
    python predict_CAID.py -f data/processed/caid3/binding.fasta
"""
import argparse
import numpy as np
import pandas as pd
import torch as tr
from pathlib import Path
import yaml
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from src.utils import read_fasta
from src.ensemble import EnsembleModel
from src.energy import AIUPredTransformer, calculate_energy_embedding

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
        '--embedding_dir', '-e',
        type=str,
        default='data/embeddings/',
        help='Directory containing precomputed pLM embeddings.'
    )
    parser.add_argument(
        '--output_dir', '-o',
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
        default=None,
        help='Cap the number of CPU threads torch uses (torch.set_num_threads).'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    return parser.parse_args()


def load_plm_embedding(acc, plm_emb_dir):
    """Load the precomputed pLM embedding for a protein accession."""
    plm_path = Path(plm_emb_dir) / f"{acc}.npy"
    if not plm_path.exists():
        raise FileNotFoundError(f"pLM embedding not found: {plm_path}")
    return np.load(plm_path)


def load_energy_model(weights_path, device):
    """Load the AIUPred model used to generate energy embeddings on the fly."""
    energy_model = AIUPredTransformer()
    energy_model.load_state_dict(tr.load(weights_path, map_location=device))
    energy_model.to(device)
    energy_model.eval()
    return energy_model


def build_ensemble_dirs(config: dict):
    """
    Build the list of ensemble model directories.

    The current layout stores one trained model per fold under a common base
    directory, e.g.:

        models/first_proposal/fold0/
        models/first_proposal/fold1/

    Each fold directory must contain both config.yaml and weights.pk.
    """
    main_model_dir = Path(config.get("main_model_dir", "models/first_proposal"))

    if not main_model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {main_model_dir}")

    fold_dirs = []
    for fold_dir in sorted(main_model_dir.iterdir(), key=lambda p: p.name):
        if not fold_dir.is_dir() or not fold_dir.name.startswith("fold"):
            continue

        cfg_path = fold_dir / "config.yaml"
        weights_path = fold_dir / "weights.pk"
        if cfg_path.exists() and weights_path.exists():
            fold_dirs.append(fold_dir)

    if not fold_dirs:
        raise ValueError(
            f"No fold directories with config.yaml and weights.pk found under {main_model_dir}"
        )

    return fold_dirs


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
        print(f"\nEmbedding shape: {emb.shape}")
        print(f"Protein ID: {acc}")
        print(f"Sequence length: {emb.shape[1]} residues")

    centers, predictions = model.pred_sliding_window(emb, step=window_step)

    elapsed_ms = int((time.time() - start_time) * 1000)
    scores = predictions[:, 1]
    labels = (scores > threshold).astype(int)
    return {
        'protein_id': acc,
        'rows': format_caid_rows(centers, sequence, scores, labels),
        'elapsed_ms': elapsed_ms,
    }


def format_caid_rows(centers, sequence, scores, labels):
    """Build (pos, aa, score, label) tuples for one protein."""
    rows = []
    for idx, score, label in zip(centers, scores, labels):
        idx = int(idx)
        aa = sequence[idx]         # centers are 0-based
        pos = idx + 1              # 1-based position for output
        rows.append((pos, aa, float(score), int(label)))
    return rows


def write_caid_block(out, acc, rows):
    out.write(f">{acc}\n")
    for pos, aa, score, label in rows:
        out.write(f"{pos}\t{aa}\t{score:.3f}\t{label}\n")


def save_protein_prediction_caid(acc: str, rows: list, output_dir: Path) -> Path:
    """Save predictions for a single protein to a CAID format file."""
    predictions_caid = output_dir / f"{acc}.caid"
    with open(predictions_caid, 'w') as out:
        write_caid_block(out, acc, rows)
    return predictions_caid


def save_combined_predictions(all_rows: list, dir: Path, save_csv: bool = False):
    """Write combined CSV and CAID files from in-memory prediction results."""
    if not all_rows:
        return None, None
    combined_caid = dir / "all_predictions.caid"

    if save_csv:
        combined_csv = dir / "all_predictions.csv"
        csv_rows = []

    with open(combined_caid, 'w') as out:
        for result in all_rows:
            write_caid_block(out, result['protein_id'], result['rows'])
            if save_csv:
                csv_rows.extend(
                    {'protein_id': result['protein_id'], 
                     'position': pos, 'aa': aa, 
                     'score': score, 'label': label}
                    for pos, aa, score, label in result['rows']
                )
    if save_csv:
        pd.DataFrame(csv_rows).to_csv(combined_csv, index=False)
    return combined_caid


def save_prediction_timings(timings: list, dir: Path, model_name: str = "emb2bind") -> Path:
    """Save per-sequence prediction timings to a CSV file."""
    if not timings:
        return None
    
    time_str = f"{datetime.now(ZoneInfo('UTC')).strftime('%a %b %e %H:%M:%S %Z %Y')}"

    timings_csv = dir / "timings.csv"

    with open(timings_csv, 'w') as f:
        f.write(f"# Running {model_name}, started {time_str}\n")
        f.write("sequence,milliseconds\n")
        for seq_id, ms in timings:
            f.write(f"{seq_id},{ms}\n")

    return timings_csv


def main():
    args = parser()

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

    # output_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    timings = []
    for header, sequence in fasta_records.items():
        acc_from_header = header.split()[0]
        if verbose:
            print(f"\nProcessing protein: {acc_from_header}")

        result = run_for_protein(
            acc_from_header,
            sequence,
            model,
            plm_emb_dir,
            energy_model,
            device,
            threshold=threshold,
        )

        if result is None:
            timings.append((acc_from_header, 0))
            continue
        
        save_protein_prediction_caid(result['protein_id'], result['rows'], output_dir)
        all_rows.append(result)
        timings.append((acc_from_header, result['elapsed_ms']))

    print(f"\nAll individual outputs saved to: {output_dir}")

    # write combined outputs (CSV and CAID-style) if we collected rows
    combined_caid = save_combined_predictions(all_rows, output_dir)
    print(f"Combined predictions saved to: {combined_caid}")

    timings_csv = save_prediction_timings(timings, output_dir)
    if timings_csv is not None:
        print(f"Timings written to: {timings_csv}")
    

if __name__ == '__main__':
    main()
