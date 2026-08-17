from __future__ import annotations

import time
from typing import Any

from workflows.model_base import BaseDeepfakeModel, validate_prediction_dict


def _clean_name(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)[:180]


def _label_matches(label: str, patterns: list[str]) -> bool:
    normalized = _clean_name(label).lower()
    return any(pattern in normalized for pattern in patterns)


class HuggingFacePipelineModel(BaseDeepfakeModel):
    def __init__(self, model_key: str, config: dict[str, Any]) -> None:
        super().__init__(model_key, config)
        self.classifier = None

    def load(self) -> None:
        from transformers import pipeline

        device = self.config.get("resolved_device", self.device)
        self.classifier = pipeline("image-classification", model=self.checkpoint_path, device=device)
        print(f"id2label: {getattr(self.classifier.model.config, 'id2label', {})}")

    def predict_batch(self, image_paths: list[str]) -> list[dict[str, Any]]:
        if self.classifier is None:
            raise RuntimeError("Model adapter has not been loaded")

        normalization = self.config.get("normalization", {}) or {}
        function_to_apply = normalization.get("function_to_apply", "sigmoid")
        start_time = time.perf_counter()
        outputs = self.classifier(
            image_paths,
            batch_size=self.batch_size,
            top_k=None,
            function_to_apply=function_to_apply,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        per_image_ms = elapsed_ms / max(len(image_paths), 1)
        normalized_outputs = self._normalize_pipeline_outputs(outputs)
        return [
            validate_prediction_dict(self._standardize_output(output, per_image_ms))
            for output in normalized_outputs
        ]

    @staticmethod
    def _normalize_pipeline_outputs(outputs: Any) -> list[list[dict[str, Any]]]:
        if not isinstance(outputs, list):
            outputs = [outputs]
        normalized: list[list[dict[str, Any]]] = []
        for output in outputs:
            if isinstance(output, dict):
                normalized.append([output])
            else:
                normalized.append(list(output))
        return normalized

    def _standardize_output(self, output: list[dict[str, Any]], inference_ms: float) -> dict[str, Any]:
        fake_probability = self._extract_fake_probability(output)
        real_probability = 1.0 - fake_probability
        prediction = "fake" if fake_probability >= self.threshold else "real"
        return {
            "fake_probability": fake_probability,
            "real_probability": real_probability,
            "prediction": prediction,
            "threshold": self.threshold,
            "inference_ms": inference_ms,
        }

    def _extract_fake_probability(self, output: list[dict[str, Any]]) -> float:
        if not output:
            raise ValueError("Empty model output")

        normalization = self.config.get("normalization", {}) or {}
        fake_patterns = normalization.get("fake_label_patterns", ["fake", "label_0"])
        real_patterns = normalization.get("real_label_patterns", ["real", "label_1"])
        by_label = {str(item.get("label", "")).lower(): float(item.get("score", 0.0)) for item in output}

        fake_scores = [
            float(item.get("score", 0.0))
            for item in output
            if _label_matches(str(item.get("label", "")), fake_patterns)
        ]
        if fake_scores:
            return max(fake_scores)

        real_scores = [
            float(item.get("score", 0.0))
            for item in output
            if _label_matches(str(item.get("label", "")), real_patterns)
        ]
        if real_scores and len(output) <= 2:
            return 1.0 - max(real_scores)

        if len(output) == 1:
            return float(output[0].get("score", 0.0))

        if "label_0" in by_label:
            return by_label["label_0"]

        raise ValueError(f"Cannot identify fake label from model output labels: {[item.get('label') for item in output]}")
