from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import onnxruntime as ort  # optional; install if you use ONNX
except ImportError:
    ort = None

class ModelRunner:
    """
    Thin wrapper around a model weights file. Replace predict() with real logic.
    """
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.model_id: Optional[str] = None
        self.session = None  # type: ignore

    def load(self, model_id: str) -> None:
        weights = self._weights_path(model_id)
        if not weights.exists():
            raise FileNotFoundError(f"weights not found: {weights}")
        if ort is None:
            self.model_id = model_id
            self.session = "stub"
            return
        self.model_id = model_id
        self.session = ort.InferenceSession(str(weights))

    def predict(self, rgb_path: Path, event_path: Optional[Path]) -> Dict[str, Any]:
        """
        Return predictions for a single tick. Replace with real preprocessing + inference.
        """
        _ = (rgb_path, event_path)
        return {
            "model_id": self.model_id,
            "detections": [],
        }

    def _weights_path(self, model_id: str) -> Path:
        """
        Find the stored weights file for a model_id. We prioritize common names
        but fall back to the first file in the directory so uploads with custom
        extensions still work.
        """
        root = self.models_dir / model_id
        preferred = ("model.onnx", "model.pt", "weights.onnx", "weights.pt")
        for name in preferred:
            cand = root / name
            if cand.exists():
                return cand
        for cand in root.iterdir():
            if cand.is_file():
                return cand
        return root / "model.onnx"
