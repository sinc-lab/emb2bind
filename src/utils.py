"""
FASTA file reading utilities for protein sequences and annotations.
"""
from pathlib import Path
from typing import Dict, Union, Optional, Any
import yaml


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


class ConfigLoader:
    def __init__(self, 
                 model_path: str = 'config/base.yaml', 
                 env_path: str = 'config/env.yaml'):
        """
        Initializes the ConfigLoader class with paths to model and environment 
        configuration files.

        Args:
            model_path: Path to the model config YAML file.
            env_path: Path to the environment config YAML file.
        """
        self.model_path = model_path
        self.env_path = env_path
        
        self.config = None # Combined configuration dictionary
        self.model = None  # Model configuration dictionary

    def load(self) -> dict:
        """
        Loads a model configuration and merges it with environment-specific 
        settings.
        Returns:
            dict: Combined configuration dictionary.
        """
        self.model = self._load_yaml(self.model_path)
        env = self._load_yaml(self.env_path)
        self.config = {**self.model, **env}
        return self.config
    
    def save(self, path: str):
        """
        Saves the model dict to a YAML file.
        Args:
            path: Path to save the configuration file.
        """
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load() first.")
        file = Path(path) / "config.yaml"
        with open(file, 'w') as f:
            yaml.dump(self.model, f, default_flow_style=False)
        print(f"Configuration saved to {file}")

    def update(self, new_config: dict):
        """
        Updates the current configuration with a new configuration dictionary.
        Args:
            new_config (dict): New configuration dictionary to merge with the existing one.
        """
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load() first.")
        self.config.update(new_config)
        self.model.update(new_config)
        print("Configuration updated.")    

    def get_config(self) -> dict:
        """
        Returns the loaded configuration dictionary.
        Returns:
            dict: The loaded configuration dictionary.
        """
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load() first.")
        return self.config

    @staticmethod
    def _load_yaml(path: str) -> dict:
        """
        Loads a YAML file from the given path and returns its content as 
        a dictionary
        """
        with open(path, 'r') as f:
            return yaml.safe_load(f)