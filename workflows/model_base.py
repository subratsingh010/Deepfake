from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDeepfakeModel(ABC):
    def __init__(self, model_key: str, config: dict[str, Any]) -> None:
        self.model_key = model_key
        self.config = config
        self.checkpoint_path = str(config["checkpoint_path"])
        self.threshold = float(config["threshold"])
        self.batch_size = int(config["batch_size"])
        self.device = config.get("device", "auto")

    @abstractmethod
    def load(self) -> None:
        """Load model weights/resources."""

    @abstractmethod
    def predict_batch(self, image_paths: list[str]) -> list[dict[str, Any]]:
        """Return standardized prediction dicts for every input path."""


def validate_prediction_dict(prediction: dict[str, Any]) -> dict[str, Any]:
    required = {"fake_probability", "real_probability", "prediction", "threshold", "inference_ms"}
    missing = required - set(prediction)
    if missing:
        raise ValueError(f"Model adapter returned incomplete prediction dict, missing: {sorted(missing)}")

    fake_probability = float(prediction["fake_probability"])
    real_probability = float(prediction["real_probability"])
    threshold = float(prediction["threshold"])
    label = str(prediction["prediction"])
    inference_ms = float(prediction["inference_ms"])

    if not 0.0 <= fake_probability <= 1.0:
        raise ValueError(f"fake_probability out of range: {fake_probability}")
    if not 0.0 <= real_probability <= 1.0:
        raise ValueError(f"real_probability out of range: {real_probability}")
    if label not in {"real", "fake"}:
        raise ValueError(f"prediction must be real or fake, got: {label}")

    return {
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "prediction": label,
        "threshold": threshold,
        "inference_ms": inference_ms,
    }
