"""
Generate protein language model embeddings for given sequences.

Usage:
    python compute_embeddings.py --fasta <path_to_fasta> --output-dir <output_directory> [--device <device>] [--skip-existing]

For example:
    python compute_embeddings.py --fasta data/samples.fasta
"""
import argparse
import logging
from pathlib import Path
from Bio import SeqIO
from src import plms

MODEL = "esm2_t6_8M_UR50D"
MAX_LENGTH = 4000

def parser():
    parser = argparse.ArgumentParser(
        description="Generate protein language model embeddings for given sequences."
        )
    parser.add_argument(
        "--fasta", 
        type=str, 
        required=True, 
        help="Path to the input FASTA file.")
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("data/embeddings/"),
        help="Directory to save the embeddings (default: data/embeddings/).")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run inference on, e.g. cuda, cuda:0, cpu (default: cuda).")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip computation entirely if every protein in the FASTA already has an embedding file.")
    
    return parser.parse_args()

def setup_logging(output_dir: Path):
    logfile = output_dir / "embeddings.log"
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",
    )

def main():
    args = parser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    logging.info(f"Generate embeddings for {MODEL}")
    
    with open(args.fasta, "r", encoding="utf-8") as handle:
        protein_ids = [record.id for record in SeqIO.parse(handle, "fasta")]

    # Check if embeddings exist
    existing_ids = {f.name.split(".")[0] for f in output_dir.iterdir() if f.is_file() and f.suffix == ".npy"}
    logging.info(f"Proteins in FASTA: {len(protein_ids)}")
    logging.info(f"Existing embeddings: {len(existing_ids)}")

    if args.skip_existing and set(protein_ids).issubset(existing_ids):
        logging.info(f"All embeddings for {MODEL} already exist, skipping.")
        return
    
    logging.info(f"Computing embeddings for {MODEL}")
    plms.get_esm2(
        fasta_path=str(args.fasta),
        output_dir=str(output_dir),
        model_name=MODEL,
        device=args.device,
        max_length=MAX_LENGTH,
    )

    logging.info(f"Embeddings for {MODEL} saved to {output_dir}")

if __name__ == "__main__":
    main()