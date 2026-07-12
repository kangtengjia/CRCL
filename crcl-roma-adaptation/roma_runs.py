from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple


@dataclass(frozen=True)
class RunPreset:
    epochs: int
    learning_rate: float
    lr_update: int
    batch_size: int


RUN_PRESETS = {
    ("scenedepict", "bigru"): RunPreset(1000, 3e-4, 20, 8),
    ("scenedepict", "bert"): RunPreset(45, 3e-4, 20, 8),
    ("scanrefer", "bigru"): RunPreset(300, 3e-4, 14, 8),
    ("scanrefer", "bert"): RunPreset(30, 3e-4, 14, 8),
    ("nr3d", "bigru"): RunPreset(800, 3e-4, 14, 8),
    ("nr3d", "bert"): RunPreset(30, 3e-4, 14, 8),
    ("3dllm", "bigru"): RunPreset(1000, 3e-5, 14, 8),
    ("3dllm", "bert"): RunPreset(30, 3e-4, 14, 8),
}


def stage_epoch_totals(total_epochs: int) -> Tuple[int, int, int, int]:
    if total_epochs < 8:
        raise ValueError("four CRCL stages require at least eight epochs")
    first = round(total_epochs * 7 / 53)
    first = max(first, 2)
    totals = (first, first, first, total_epochs - 3 * first)
    if totals[-1] < 2:
        raise ValueError("final CRCL stage requires at least two epochs")
    return totals


def format_markdown_metrics(dataset: str, text_encoder: str, metrics: Mapping[str, float]) -> str:
    headers = ["Dataset", "Text", "R@1", "R@5", "R@10", "R@30", "Rsum", "MedR", "MeanR", "MRR"]
    values = [
        dataset,
        text_encoder,
        *(f"{float(metrics[key]):.2f}" for key in ["R@1", "R@5", "R@10", "R@30", "Rsum", "MedR", "MeanR", "MRR"]),
    ]
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n| " + " | ".join(values) + " |\n"
