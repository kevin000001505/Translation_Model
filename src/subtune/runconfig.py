"""The settings of one training run, as saved to run.json and to the model's metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class TrainConfig:
    model_id: str
    family: str
    src: str
    tgt: str
    method: str = "LoRA"  # LoRA | QLoRA | Full
    target: str = "local"  # local | modal
    gpu: str = ""  # Modal GPU type
    multimodal: bool = False
    trust_remote_code: bool = False
    epochs: float = 1
    max_pairs: int = 0  # 0 = all
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    batch_size: int = 8
    grad_accum: int = 4
    max_seq_len: int = 256
    eval_sentences: int = 200
    seed: int = 42
    # Selected titles: title key -> feature ids. Only the ready (aligned) ones are used.
    titles: dict[str, list[int]] = field(default_factory=dict)
    # Run "download what the quota allows" before training.
    prepare_first: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def hyperparameters(self) -> dict:
        keys = ("epochs", "max_pairs", "learning_rate", "lora_r", "lora_alpha", "lora_dropout", "batch_size",
                "grad_accum", "max_seq_len", "eval_sentences", "seed")
        d = {k: getattr(self, k) for k in keys}
        if self.method == "Full":
            for k in ("lora_r", "lora_alpha", "lora_dropout"):
                d.pop(k)
        return d
