"""ML Strategy Policy: neural network that outputs intent, target, and risk.

Architecture: simple MLP (fast inference, <2ms on CPU).
Falls back gracefully if PyTorch is not installed.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.types import Intent, StrategyOutput, Vec3

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Provide minimal stubs so the module is importable without torch
    torch = None  # type: ignore
    nn = None  # type: ignore

OBS_DIM = 34
NUM_INTENTS = Intent.count()  # 7


def _make_strategy_network():
    """Create StrategyNetwork class (requires torch)."""
    if not TORCH_AVAILABLE:
        return None

    class _Net(nn.Module):
        """MLP policy network for strategy decisions."""

        def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 128):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.intent_head = nn.Linear(hidden, NUM_INTENTS)
            self.target_head = nn.Linear(hidden, 3)
            self.risk_head = nn.Linear(hidden, 1)

        def forward(self, obs):
            h = self.shared(obs)
            intent_logits = self.intent_head(h)
            target = self.target_head(h)
            risk = torch.sigmoid(self.risk_head(h))
            return intent_logits, target, risk

    return _Net


# The actual class (or None if no torch)
StrategyNetwork = _make_strategy_network()


class MLStrategy:
    """Wrapper that loads a trained model and produces StrategyOutput."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self._loaded = False

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available — ML strategy disabled")
            return

        if model_path:
            self.load(model_path)

    def load(self, path: str) -> bool:
        """Load a trained model from disk with checksum validation."""
        if not TORCH_AVAILABLE or StrategyNetwork is None:
            return False

        p = Path(path)
        if not p.exists():
            logger.warning(f"Model file not found: {path}")
            return False

        try:
            checkpoint = torch.load(p, map_location="cpu", weights_only=False)

            if "checksum" in checkpoint:
                state_bytes = str(checkpoint["state_dict"]).encode()
                computed = hashlib.md5(state_bytes).hexdigest()
                if computed != checkpoint["checksum"]:
                    logger.error(f"Model checksum mismatch: {path}")
                    return False

            self.model = StrategyNetwork()
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            self._loaded = True
            logger.info(f"Loaded ML strategy model from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self._loaded = False
            return False

    @property
    def is_ready(self) -> bool:
        return self._loaded and self.model is not None

    def decide(self, obs_tensor: np.ndarray) -> Optional[StrategyOutput]:
        """Run inference on an observation tensor."""
        if not self.is_ready or not TORCH_AVAILABLE:
            return None

        try:
            with torch.no_grad():
                obs = torch.from_numpy(obs_tensor).unsqueeze(0)
                intent_logits, target, risk = self.model(obs)

                intent_idx = int(intent_logits.argmax(dim=-1).item())
                intent = Intent(intent_idx)

                t = target.squeeze(0).numpy()
                target_pos = Vec3(
                    float(t[0]) * 5120.0,
                    float(t[1]) * 5120.0,
                    max(0.0, float(t[2]) * 2000.0),
                )

                risk_val = float(risk.item())
                return StrategyOutput(intent=intent, target=target_pos, risk=risk_val)

        except Exception as e:
            logger.error(f"ML inference failed: {e}")
            return None

    @staticmethod
    def save_checkpoint(model, path: str, metadata: Optional[dict] = None) -> None:
        """Save a model checkpoint with checksum."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required to save models")

        state_dict = model.state_dict()
        state_bytes = str(state_dict).encode()
        checksum = hashlib.md5(state_bytes).hexdigest()

        checkpoint = {
            "state_dict": state_dict,
            "checksum": checksum,
            "obs_dim": OBS_DIM,
            "num_intents": NUM_INTENTS,
        }
        if metadata:
            checkpoint["metadata"] = metadata

        torch.save(checkpoint, path)
        logger.info(f"Saved model checkpoint to {path} (checksum: {checksum})")
