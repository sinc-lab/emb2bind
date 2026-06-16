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
import sys
import time
from datetime import datetime

sys.path.append(str(Path.cwd()))
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
        '--output-dir', '-o',
        type=str,
        default='results/',
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
        default='cuda',
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
    """Build the list of full ensemble model directories from the YAML config.

    Expected config keys:
        - main_model_dir: base directory that contains the fold folders
        - subdir: training subdirectory inside each fold
        - ensemble_dirs: list of run-directory names, one per fold

    The function returns paths like:
        main_model_dir/fold_0/subdir/run_name
    """
    main_model_dir = Path(config["main_model_dir"])
    subdir = config["subdir"]
    ensemble_runs = config.get("ensemble_dirs", [])

    if isinstance(ensemble_runs, dict):
        # Allow future configs to store explicit fold -> run mappings.
        items = ensemble_runs.items()
        return [main_model_dir / fold_name / subdir / run_name for fold_name, run_name in items]

    if not ensemble_runs:
        raise ValueError("'ensemble_dirs' must contain at least one model directory name")

    return [main_model_dir / f"fold_{i}" / subdir / run_name for i, run_name in enumerate(ensemble_runs)]


def build_embedding(
        acc,
        sequence,
        plm_emb_dir="data/embeddings/plm/esm2_6m/", # ! UPDATE
        energy_model=None,
        device='cpu',
        reduce_method= "weighted"
        ):
    """Build the combined embedding from the pLM embedding and on-the-fly energy embedding."""
    plm_emb = load_plm_embedding(acc, plm_emb_dir)       # (emb_dim_plm, L)

    energy_tensor = calculate_energy_embedding(sequence, energy_model, device, reduce=reduce_method, scale_factor=36.8080)
    energy_emb = energy_tensor.detach().cpu().numpy()    # (emb_dim_energy, L)

    emb = np.concatenate([energy_emb, plm_emb], axis=0)  # (emb_dim_energy + emb_dim_plm, L)
    return tr.tensor(emb, dtype=tr.float32)


def run_for_protein(
        acc: str,
        sequence,
        model,
        plm_emb_dir,
        energy_model,
        device,
        reduce_method='weighted',
        window_step=1,
        threshold=0.5,
        verbose=False,
        ):
    """Run prediction pipeline for a single protein accession."""
    start_time = time.time()

    try:
        emb = build_embedding(
            acc,
            sequence,
            plm_emb_dir=plm_emb_dir,
            energy_model=energy_model,
            device=device,
            reduce_method=reduce_method,
        )
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
        'acc': acc,
        'centers': np.asarray(centers),
        'predictions': predictions,
        'scores': scores,
        'labels': labels,
        'sequence': sequence,
        'elapsed_ms': elapsed_ms,
    }


def save_protein_prediction_caid(acc, centers, scores, labels, sequence, output_dir): # !
    """Save predictions for a single protein to a CAID format file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_caid = output_dir / f"{acc}.caid"

    centers = np.asarray(centers)
    with open(predictions_caid, 'w') as out:
        out.write(f">{acc}\n")
        for idx, score, label in zip(centers, scores, labels):
            aa = sequence[idx]          # centers are 0-based
            pos = idx + 1               # 1-based position for output
            out.write(f"{pos}\t{aa}\t{score:.3f}\t{label}\n")

    return predictions_caid


def collect_prediction_row(all_rows: list, result: dict): # !
    """Collect one protein's predictions into the consolidated row buffer."""
    all_rows.append({
        'protein_id': result['acc'],
        'centers': result['centers'],
        'sequence': result['sequence'],
        'scores': result['scores'],
        'labels': result['labels'],
    })


def save_combined_predictions(all_rows: list, model_dir: Path): # !
    """Write combined CSV and CAID files from in-memory prediction results."""
    if not all_rows:
        return None, None

    combined_dir = model_dir / "predictions"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = combined_dir / "emb2bind.csv"
    combined_caid = combined_dir / "emb2bind.caid"

    rows = []
    with open(combined_caid, 'w') as out:
        for result in all_rows:
            protein_id = result['protein_id']
            centers = result['centers']
            sequence = result['sequence']
            scores = result['scores']
            labels = result['labels']

            out.write(f">{protein_id}\n") # CAID header line

            for idx, score, label in zip(centers, scores, labels):
                idx = int(idx)
                aa = sequence[idx]
                pos = idx + 1
                # CAID
                out.write(f"{pos}\t{aa}\t{score:.3f}\t{label}\n")
                # CSV row
                rows.append({
                    'protein_id': protein_id,
                    'position': pos,
                    'aa': aa,
                    'score': float(score),
                    'label': int(label),
                })

    pd.DataFrame(rows).to_csv(combined_csv, index=False)
    return combined_csv, combined_caid


def save_prediction_timings(timings: list, model_dir: Path):
    """Save per-sequence prediction timings to a CSV file."""
    if not timings:
        return None

    combined_dir = model_dir / "predictions"
    combined_dir.mkdir(parents=True, exist_ok=True)
    timings_csv = combined_dir / "timings.csv"
    with open(timings_csv, 'w') as f:
        f.write(f"# Running predict_CAID, started {datetime.now().ctime()}\n")
        f.write("sequence,milliseconds\n")
        for seq_id, ms in timings:
            f.write(f"{seq_id},{ms}\n")

    return timings_csv


def main():
    args = parser()

    verbose = args.verbose
    device = args.device
    plm_emb_dir = args.embedding_dir
    model_dir = Path(args.output_dir)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Process multiple proteins from the input FASTA
    energy_model_weights = config.get('energy_model_weights', 'data/AIUPred/embedding.pt')
    energy_model = load_energy_model(energy_model_weights, device)

    # Use an ensemble of trained models (list of model folders)
    ensemble_dirs = build_ensemble_dirs(config)
    if verbose:
        print("Ensemble model directories:")
        for model_dir in ensemble_dirs:
            print(f"  - {model_dir}")
    model = EnsembleModel(ensemble_dirs, device=device)

    fasta_records = read_fasta(args.fasta)
    if not fasta_records:
        raise ValueError(f"No sequences found in FASTA: {args.fasta}")
    threshold = config.get('threshold', 0.5)

    results = []
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
            results.append((acc_from_header, None, None))
            continue

        output_dir = model_dir / "predictions"
        predictions_caid = save_protein_prediction_caid(
            acc_from_header,
            result['centers'],
            result['scores'],
            result['labels'],
            sequence,
            output_dir,
        )
        collect_prediction_row(all_rows, result)
        timings.append((acc_from_header, result['elapsed_ms']))
        results.append((acc_from_header, None, predictions_caid))

    print(f"\n✓ All outputs saved to: {model_dir}")

    # write combined outputs (CSV and CAID-style) if we collected rows
    combined_csv, combined_caid = save_combined_predictions(all_rows, model_dir)
    if combined_csv is not None:
        print(f"\nCombined predictions saved to: {combined_csv}")
        print(f"Combined CAID-style saved to: {combined_caid}")
        
    # write timings.csv
    timings_csv = save_prediction_timings(timings, model_dir)
    if timings_csv is not None:
        print(f"Timings written to: {timings_csv}")
    

if __name__ == '__main__':
    main()
