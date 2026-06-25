import numpy as np
import torch as tr
from torch import nn
from torch.nn.functional import softmax
from pathlib import Path
from src.utils import ConfigLoader
from src.model import BaseModel


class EnsembleModel(nn.Module):
    """
    Ensemble class that implements a soft-voting of multiple trained BaseModel 
    instances. Each model is loaded from a folder containing the config and 
    weights. The forward method averages the predicted probabilities from all models.
    """
    def __init__(self, model_dirs, device='cpu'):
        super().__init__()
        self.device = device
        self.models = nn.ModuleList()
        self._model_configs = []
        self._model_window_len = []

        for md in model_dirs:
            md_path = Path(md)
            cfg_path = md_path / 'config.yaml'
            weights_path = md_path / 'weights.pk'
            if not cfg_path.exists() or not weights_path.exists():
                raise FileNotFoundError(
                    f"Model folder must contain config.yaml and weights.pk: {md}"
                    )

            loader = ConfigLoader(model_path=str(cfg_path))
            model_cfg = loader.load()
            self._model_configs.append(model_cfg)
            self._model_window_len.append(model_cfg['win_len'])

            # Create individual BaseModel instance
            model = BaseModel(
                nclasses=len(model_cfg['categories']),
                lr=model_cfg['lr'],
                device=device,
                filters=model_cfg['filters'],
                kernel_size=model_cfg['kernel_size'],
                num_layers=model_cfg['n_resnet'],
                dropout_rate=model_cfg['dropout_rate'],
            )

            # Load trained weights
            state = tr.load(str(weights_path), map_location=device)
            model.load_state_dict(state)
            model.eval()
            self.models.append(model)

    def forward(self, x):
        """
        Run input through all models and return averaged probabilities.

        Args:
            x: input tensor with shape (batch_size, emb_size, win_size).
        Returns:
            Tensor of averaged probabilities (batch_size, nclasses).
        """
        probs = []
        for m in self.models:
            logits = m(x)
            p = softmax(logits, dim=1)
            probs.append(p)

        stacked = tr.stack(probs, dim=0)  # (n_members, batch, nclasses)
        avg = tr.mean(stacked, dim=0)     # (batch, nclasses)
        return avg

    def pred_sliding_window(self, emb, step=1):
        """
        Predict probabilities for each position in the input embedding using a 
        sliding window approach. The window is centered on each position and 
        the model predicts the probability of binding for that window.

        Args:
            emb: input embedding tensor with shape (emb_size, seq_len).
            step: step size for sliding the window.
        Returns:
            centers: list of center positions for each window.
            pred: tensor of predicted probabilities for each window.
        """
        L = emb.shape[1]
        emb_size = emb.shape[0]
        centers = np.arange(0, L, step)

        probs = np.empty((len(self.models), len(centers), 2))

        for (i, m) in enumerate(self.models):

            window_len = self._model_window_len[i]

            batch = tr.zeros((len(centers), emb_size, window_len), dtype=tr.float)

            for k, center in enumerate(centers):
                start = max(0, center - window_len // 2)
                end = min(L, center + window_len // 2)
                batch[k, :, :end - start] = emb[:, start:end].unsqueeze(0)

            with tr.no_grad():
                logits = m(batch.to(self.device))
                pred = softmax(logits, dim=1).cpu().detach()
                probs[i] = pred.numpy()

        stacked = np.stack(probs, axis=0)  # (n_members, n_windows, nclasses)
        avg = np.mean(stacked, axis=0)     # (n_windows, nclasses)

        return centers, avg


def build_ensemble_dirs(config: dict):
    """
    Build a list of ensemble model directories.
    One trained model is stored per fold under a common base directory, e.g.:
        - models/fold0/
        - models/fold1/
        ...
    Each fold directory must contain both config.yaml and weights.pk.
    """
    main_model_dir = Path(config["main_model_dir"])

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