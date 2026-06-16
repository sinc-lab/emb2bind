"""
Energy embedding module using Transformer network.
Generates residue-level energy embeddings from protein sequences.
"""
import torch as tr
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import pad
import math

AA_CODE = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
# Non-standard amino acids (B, Z, J, U, O) will be replaced with 'X' 
WINDOW = 100 
VERBOSE = False

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the Transformer network
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = tr.zeros(max_len, d_model)
        position = tr.arange(0, max_len, dtype=tr.float).unsqueeze(1)
        div_term = tr.exp(tr.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = tr.sin(position * div_term)  # Even indices
        pe[:, 1::2] = tr.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input"""
        return x + self.pe[:, :x.size(1), :]


class AIUPredTransformer(nn.Module):
    """
    Transformer model for energy prediction.
    Generates per-residue energy embeddings.
    """
    def __init__(self):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = 32  # Embedding dimension
        self.pos_encoder = PositionalEncoding(self.d_model)
        encoder_layers = TransformerEncoderLayer(self.d_model, 2, 64, 0, batch_first=True)
        # d_model=32, nhead=2, dim_feedforward=64, dropout=0
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)  # 2 layers
        self.encoder = nn.Embedding(21, self.d_model)
        self.decoder = nn.Linear((WINDOW + 1) * self.d_model, 1)

    def forward(self, src: Tensor, embed_only=False) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)  # (L x Window+1 x Embed_dim)
        embedding = self.transformer_encoder(src)
        if embed_only:
            return embedding
        output = tr.flatten(embedding, 1)
        output = self.decoder(output)
        return tr.squeeze(output)


@tr.no_grad()
def tokenize(sequence, device):
    """
    Tokenize an amino acid sequence. Non-standard amino acids are treated as X.
    Args:
        sequence: Amino acid sequence in string
        device: Device to run on (CUDA{x} or CPU)
    Returns:
        Tokenized tensor
    """
    return tr.tensor(
        [AA_CODE.index(aa) if aa in AA_CODE else 20 for aa in sequence],
        device=device
    )


def clean_sequence(sequence: str) -> tuple[str, int]:
    """
    Replace ambiguous amino-acid letters with 'X'. Returns cleaned sequence and the
    number of replacements performed.

    Replacements done: B, Z, J, U, O -> X. 
    """
    if sequence is None:
        return sequence, 0
    
    ambiguous = set(['B', 'Z', 'J', 'U', 'O'])

    in_seq = sequence.strip().upper()
    replaced = 0
    out_seq = []
    for aa in in_seq:
        if aa in ambiguous:
            out_seq.append('X')
            replaced += 1
        else:
            out_seq.append(aa)

    return ''.join(out_seq), replaced


def calculate_energy_embedding(sequence, energy_model, device, reduce=None, scale_factor=1.0):
    """
    Calculates residue energy embedding from a sequence using a transformer
    network.
    
    Args:
        sequence: Amino acid sequence in string
        energy_model: Transformer model
        device: Device to run on (CUDA{x} or CPU)
        reduce: Reduction method - 'sum', 'mean', 'central', or None
        scale_factor: Optional scaling factor to apply to the embedding (default: 1.0)
    Returns:
        Tensor of energy embeddings per residue (L, D) or (L, WINDOW+1, D) if 
        no reduction
    """
    # Normalize sequence: replace ambiguous/reserved letters with 'X'
    sequence, replaced = clean_sequence(sequence)
    if replaced and VERBOSE:
        print(f"Warning: replaced {replaced} ambiguous amino acid(s) with 'X' in sequence")

    tokenized_sequence = tokenize(sequence, device)
    padded_token = pad(tokenized_sequence, (WINDOW // 2, WINDOW // 2), 'constant', 20)
    unfolded_tokens = padded_token.unfold(0, WINDOW + 1, 1)
    embed_only = True if reduce is not None else False
    embedding = energy_model(unfolded_tokens, embed_only=embed_only)  # (L, WINDOW+1, D)

    # Reduce embedding if needed, to get per-residue representation (L, D)
    if reduce == 'sum':
        embedding = embedding.sum(dim=1)

    elif reduce == 'mean':
        embedding = embedding.mean(dim=1)

    elif reduce == 'central':
        central_idx = WINDOW // 2
        embedding = embedding[:, central_idx, :]
        
    elif reduce == 'weighted':
        central_idx = WINDOW // 2        
        window_size = WINDOW + 1
        sigma = window_size / 6  # Proposed standard deviation (aprox. 16.67 for WINDOW=100)
        
        # Create Gaussian weights centered at the middle position
        positions = tr.arange(window_size, device=embedding.device, dtype=tr.float32)
        weights = tr.exp(-0.5 * ((positions - central_idx) / sigma) ** 2)
        weights = weights / weights.sum()  # Normalize to sum to 1
        
        # Apply weights: (L, WINDOW+1, D) * (WINDOW+1,) -> (L, D)
        embedding = (embedding * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)

    # We need the embedding to be (D, L)
    embedding = embedding.T
    
    # Apply scaling if factor is provided
    if scale_factor != 1.0:
        embedding = embedding * scale_factor

    return embedding
