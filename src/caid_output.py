import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


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


def save_combined_predictions(all_rows: list, dir: Path, save_csv: bool = False,
                              file_name: str = "all_predictions"):
    """Write combined CSV and CAID files from in-memory prediction results."""
    if not all_rows:
        return None, None
    combined_caid = dir / f"{file_name}.caid"

    if save_csv:
        combined_csv = dir / f"{file_name}.csv"
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
