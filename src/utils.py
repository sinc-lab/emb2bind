"""
FASTA file reading utilities for protein sequences and annotations.
"""
from pathlib import Path
from typing import Dict, Union, Optional, Any


def read_fasta(file_location: Union[str, Path], 
               data_type: type = str, 
               split: Optional[str] = None) -> Dict[str, Any]:
    """
    Generic FASTA reader that handles multi-line sequences and specific data types.
    
    :param file_location: Path to the FASTA file
    :param data_type: The type to convert sequence data into (default: str)
    :param split: Delimiter if data needs splitting (e.g., space separated numbers)
    :return: Dictionary of headers and parsed sequences
    """
    file_path = Path(file_location)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    fasta = {}
    header = None
    # Use a list to accumulate sequence parts (faster than string concatenation)
    sequence_buffer = []

    def save_buffer(hdr, buf):
        if hdr and buf:
            if data_type == str:
                fasta[hdr] = "".join(buf)
            else:
                # If non-string, flatten the buffer list
                flat_list = [item for sublist in buf for item in sublist]
                fasta[hdr] = flat_list

    with file_path.open('r') as fn:
        for line in fn:
            line = line.strip()
            if not line: continue

            if line.startswith('>'):
                # Save previous entry before starting new one
                save_buffer(header, sequence_buffer)
                
                header = line.lstrip('>')
                sequence_buffer = []
            else:
                if header is None:
                    continue # Skip content before first header
                
                if data_type == str:
                    sequence_buffer.append(line)
                else:
                    # Handle non-string data types (e.g., numeric scores)
                    parts = line.split(split)
                    try:
                        converted = [data_type(x) for x in parts if x]
                        sequence_buffer.append(converted)
                    except ValueError:
                         print(f"Warning: Could not convert data to {data_type} for {header}")

        # Save the last entry
        save_buffer(header, sequence_buffer)

    return fasta