from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch


@dataclass(frozen=True)
class ResumeState:
    stage_index: int
    stage_epoch: int
    global_epoch: int
    move_gt: object
    best_score: float
    best_metrics: Dict[str, float]
    iterations: int


def build_checkpoint(*, stage_index: int, stage_epoch: int, global_epoch: int, model_state: object, optimizer: torch.optim.Optimizer, move_gt: object, best_score: float, best_metrics: Mapping[str, float], iterations: int, config: Mapping[str, object], input_manifest: Mapping[str, object] | None = None) -> Dict[str, object]:
    return {
        "format_version": 2, "stage_index": stage_index, "stage_epoch": stage_epoch,
        "global_epoch": global_epoch, "model": model_state, "optimizer": optimizer.state_dict(),
        "move_gt": move_gt, "best_score": float(best_score),
        "best_metrics": {key: float(value) for key, value in best_metrics.items()},
        "iterations": iterations, "config": dict(config), "input_manifest": dict(input_manifest or {}),
    }


def restore_checkpoint(checkpoint: Mapping[str, object], optimizer: torch.optim.Optimizer, *, weights_only: bool = False) -> ResumeState:
    if int(checkpoint.get("format_version", 0)) < 2:
        if not weights_only:
            raise ValueError("legacy checkpoint lacks full resume state")
        return ResumeState(0, 0, 0, None, 0.0, {}, 0)
    optimizer_state = checkpoint.get("optimizer")
    if not isinstance(optimizer_state, dict):
        raise ValueError("checkpoint is missing optimizer state")
    optimizer.load_state_dict(optimizer_state)
    raw_metrics = checkpoint.get("best_metrics", {})
    if not isinstance(raw_metrics, Mapping):
        raise ValueError("checkpoint best_metrics must be a mapping")
    return ResumeState(
        int(checkpoint["stage_index"]), int(checkpoint["stage_epoch"]), int(checkpoint["global_epoch"]),
        checkpoint.get("move_gt"), float(checkpoint.get("best_score", 0.0)),
        {key: float(value) for key, value in raw_metrics.items()}, int(checkpoint.get("iterations", 0)),
    )
