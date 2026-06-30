"""Configuration system — YAML config with CLI overrides."""

import yaml
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-3B-Instruct"
    dtype: str = "float16"


@dataclass
class DatasetConfig:
    train_path: str = "data/train.jsonl"
    test_path: str = "data/test.jsonl"
    max_samples: Optional[int] = None


@dataclass
class LoraConfig:
    rank: int = 256
    alpha: int = 512
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class TrainingConfig:
    max_steps: int = 200
    learning_rate: float = 2e-5
    warmup_steps: int = 20
    batch_size: int = 1
    gradient_accumulation_steps: int = 2
    optimizer: str = "adamw_bnb_8bit"
    gradient_checkpointing: bool = True
    bf16: bool = True
    max_length: int = 2048
    logging_steps: int = 10
    seed: int = 42


@dataclass
class EvaluationConfig:
    quality_gate: float = 0.95
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1"
    ])
    max_test_samples: int = 500


@dataclass
class ExportConfig:
    quantize: Optional[str] = None
    format: str = "safetensors"
    benchmark: bool = True


@dataclass
class OutputConfig:
    dir: str = "outputs/model"
    save_model_card: bool = True


@dataclass
class HephaestusConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "HephaestusConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        def _make(cls_type, data):
            if data is None:
                return cls_type()
            field_types = {f.name: f.type for f in cls_type.__dataclass_fields__.values()}
            filtered = {}
            for k, v in data.items():
                if k in field_types:
                    filtered[k] = v
            return cls_type(**filtered)

        return cls(
            model=_make(ModelConfig, raw.get("model")),
            dataset=_make(DatasetConfig, raw.get("dataset")),
            lora=_make(LoraConfig, raw.get("lora")),
            training=_make(TrainingConfig, raw.get("training")),
            evaluation=_make(EvaluationConfig, raw.get("evaluation")),
            export=_make(ExportConfig, raw.get("export")),
            output=_make(OutputConfig, raw.get("output")),
        )

    @classmethod
    def from_profile(cls, profile: str) -> "HephaestusConfig":
        profiles = {
            "latency": cls(
                model=ModelConfig(name="Qwen/Qwen2.5-0.5B-Instruct"),
                lora=LoraConfig(rank=64, alpha=128),
                training=TrainingConfig(batch_size=2, max_length=512),
                export=ExportConfig(quantize="int4"),
            ),
            "balanced": cls(
                model=ModelConfig(name="Qwen/Qwen2.5-1.5B-Instruct"),
                lora=LoraConfig(rank=128, alpha=256),
                training=TrainingConfig(batch_size=2, max_length=1024),
                export=ExportConfig(quantize="int4"),
            ),
            "accuracy": cls(
                model=ModelConfig(name="Qwen/Qwen2.5-3B-Instruct"),
                lora=LoraConfig(rank=256, alpha=512),
                training=TrainingConfig(batch_size=1, max_length=2048),
                export=ExportConfig(quantize=None),
            ),
        }
        if profile not in profiles:
            raise ValueError(f"Choose: latency/balanced/accuracy")
        return profiles[profile]

    def to_dict(self):
        return asdict(self)
