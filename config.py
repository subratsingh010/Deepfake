from pathlib import Path


BENCHMARK_CONFIG = {
    "data_root": Path("/Users/subrat/Desktop/Deepfake"),
    "output_root": Path.cwd() / "output",
    "target_dimensions": ["original", 1024, 720, 512, 384, 256],
    "resize_modes": ["aspect", "square"],
    "jpeg_quality_levels": [95, 80, 60, 40],
    "checkpoint_every": 25,
    "random_seed": 42,
    "keep_generated_variants": False,
    "metadata_workers": 4,
    "balance_labels": True,
    "clean_model_output": True,
}


MODEL_CONFIGS = {
    "cf_vit": {
        "enabled": True,
        "adapter_path": "workflows.adapters.huggingface_pipeline_adapter.HuggingFacePipelineModel",
        "checkpoint_path": "buildborderless/CommunityForensics-DeepfakeDet-ViT",
        "input_size": None,
        "batch_size": 8,
        "device": "auto",
        "threshold": 0.50,
        "output_format": "huggingface_image_classification",
        "class_order": ["fake", "real"],
        "normalization": {
            "function_to_apply": "sigmoid",
            "fake_label_patterns": [
                "fake",
                "deepfake",
                "synthetic",
                "generated",
                "manipulated",
                "ai",
                "spoof",
                "label_0",
            ],
            "real_label_patterns": [
                "real",
                "authentic",
                "genuine",
                "original",
                "camera",
                "label_1",
            ],
        },
    },
}
