from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler


DATASET_ALIASES = {
    "our_data": "scenedepict",
    "3d_text_retrv": "scenedepict",
    "scenedepict-3d2t": "scenedepict",
    "3d_llm": "3dllm",
    "llm-3d-scene": "3dllm",
}


@dataclass(frozen=True)
class RoMaBundle:
    captions: List[str]
    scene_ids: List[str]
    scene_indices: List[int]
    feature_scene_ids: List[str]
    features: np.ndarray


def canonical_dataset_name(name: str) -> str:
    lowered = str(name).strip().lower()
    return DATASET_ALIASES.get(lowered, lowered)


def _split_tag(split: str) -> str:
    return "train" if split == "train" else "val"


def _json(path: Path) -> List[Dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, list):
        raise ValueError(f"expected JSON list: {path}")
    return result


def _jsonl(path: Path) -> List[Dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ordered_unique(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(values))


def source_paths(data_root: str | Path, data_name: str, data_split: str) -> List[Path]:
    root = Path(data_root)
    dataset = canonical_dataset_name(data_name)
    split = _split_tag(data_split)
    if dataset == "scanrefer":
        return [root / f"scanrefer_{split}.jsonl", root / f"pt2vec_200_random_{split}.npy"]
    if dataset == "scenedepict":
        return [root / f"3D_Text_Retrv_{split}_final_sorted.json", root / f"3D_Text_Retrv_grid_{split}.npy"]
    if dataset == "3dllm":
        return [root / f"3d_llm_scene_description_{split}_sorted.json", root / f"3d_llm_grid_{split}.npy"]
    if dataset == "nr3d":
        return [
            root / f"nr3d_{split}.jsonl", root / "scanrefer_train.jsonl", root / "scanrefer_val.jsonl",
            root / "pt2vec_200_random_train.npy", root / "pt2vec_200_random_val.npy",
            root / "3D_Text_Retrv_train_final_sorted.json", root / "3D_Text_Retrv_val_final_sorted.json",
            root / "3D_Text_Retrv_grid_train.npy", root / "3D_Text_Retrv_grid_val.npy",
        ]
    raise ValueError(f"unsupported RoMa dataset: {dataset}")


def _rows(root: Path, dataset: str, split: str) -> tuple[List[str], List[str]]:
    if dataset == "scanrefer":
        rows = _jsonl(root / f"scanrefer_{split}.jsonl")
        return [str(row["description"]) for row in rows], [str(row["scene_id"]) for row in rows]
    if dataset == "nr3d":
        rows = _jsonl(root / f"nr3d_{split}.jsonl")
        return [str(row["description"]) for row in rows], [str(row.get("scan_id") or row["scene_id"]) for row in rows]
    if dataset == "scenedepict":
        rows = _json(root / f"3D_Text_Retrv_{split}_final_sorted.json")
        return [str(row["description"]) for row in rows], [str(row["scene_id"]) for row in rows]
    if dataset == "3dllm":
        rows = _json(root / f"3d_llm_scene_description_{split}_sorted.json")
        captions = [str(row["answers"][0]) for row in rows if isinstance(row.get("answers"), list) and row["answers"]]
        if len(captions) != len(rows):
            raise ValueError("3dllm rows must contain non-empty answers")
        return captions, [str(row["scene_id"]) for row in rows]
    raise ValueError(f"unsupported RoMa dataset: {dataset}")


def _scanrefer_pool(root: Path) -> tuple[List[str], np.ndarray]:
    ids, arrays = [], []
    for split in ("train", "val"):
        rows = _jsonl(root / f"scanrefer_{split}.jsonl")
        scene_ids = _ordered_unique([str(row["scene_id"]) for row in rows])
        features = np.load(root / f"pt2vec_200_random_{split}.npy")
        if len(scene_ids) != len(features):
            raise ValueError(f"ScanRefer {split} feature scene count mismatch")
        ids.extend(scene_ids)
        arrays.append(features)
    if len(ids) != len(set(ids)):
        raise ValueError("ScanRefer train/val feature scene orders overlap")
    return ids, np.concatenate(arrays, axis=0)


def _scenedepict_pool(root: Path) -> tuple[List[str], np.ndarray]:
    ids, arrays = [], []
    for split in ("train", "val"):
        rows = _json(root / f"3D_Text_Retrv_{split}_final_sorted.json")
        scene_ids = _ordered_unique([str(row["scene_id"]) for row in rows])
        features = np.load(root / f"3D_Text_Retrv_grid_{split}.npy")
        if len(scene_ids) != len(features):
            raise ValueError(f"SceneDepict {split} feature scene count mismatch")
        ids.extend(scene_ids)
        arrays.append(features)
    return ids, np.concatenate(arrays, axis=0)


def load_roma_bundle(data_root: str | Path, data_name: str, data_split: str) -> RoMaBundle:
    root = Path(data_root)
    dataset = canonical_dataset_name(data_name)
    split = _split_tag(data_split)
    captions, scene_ids = _rows(root, dataset, split)
    feature_scene_ids = _ordered_unique(scene_ids)
    if dataset == "nr3d":
        scan_ids, scan_features = _scanrefer_pool(root)
        depict_ids, depict_features = _scenedepict_pool(root)
        feature_by_scene = {scene_id: scan_features[index] for index, scene_id in enumerate(scan_ids)}
        for index, scene_id in enumerate(depict_ids):
            feature_by_scene.setdefault(scene_id, depict_features[index])
        missing = sorted(set(feature_scene_ids) - set(feature_by_scene))
        if missing:
            raise ValueError(f"caption scenes missing from RoMa DGCNN feature pools: {missing[:5]}")
        features = np.stack([feature_by_scene[scene_id] for scene_id in feature_scene_ids])
    else:
        if dataset == "scanrefer":
            feature_path = root / f"pt2vec_200_random_{split}.npy"
        elif dataset == "scenedepict":
            feature_path = root / f"3D_Text_Retrv_grid_{split}.npy"
        elif dataset == "3dllm":
            feature_path = root / f"3d_llm_grid_{split}.npy"
        else:
            raise ValueError(f"unsupported RoMa dataset: {dataset}")
        features = np.load(feature_path)
    if features.ndim != 3 or len(feature_scene_ids) != len(features):
        raise ValueError(f"feature scene count mismatch for {dataset}/{split}: {features.shape}")
    feature_index = {scene_id: index for index, scene_id in enumerate(feature_scene_ids)}
    return RoMaBundle(captions, scene_ids, [feature_index[scene_id] for scene_id in scene_ids], feature_scene_ids, features)


class SceneUniqueBatchSampler(Sampler[List[int]]):
    def __init__(self, scene_indices: Sequence[int], batch_size: int, *, shuffle: bool = True, seed: int = 2022) -> None:
        self.scene_indices = list(scene_indices)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size <= 0 or self.batch_size > len(set(self.scene_indices)):
            raise ValueError(f"batch_size must not exceed {len(set(self.scene_indices))} unique scenes")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        grouped: Dict[int, Deque[int]] = defaultdict(deque)
        for index, scene_index in enumerate(self.scene_indices):
            grouped[scene_index].append(index)
        for indices in grouped.values():
            if self.shuffle:
                values = list(indices)
                rng.shuffle(values)
                indices.clear()
                indices.extend(values)
        active = list(grouped)
        while active:
            if len(active) < self.batch_size:
                break
            if self.shuffle:
                rng.shuffle(active)
            batch_scenes = active[:self.batch_size]
            yield [grouped[scene].popleft() for scene in batch_scenes]
            active = [scene for scene in active if grouped[scene]]

    def __len__(self) -> int:
        return len(self.scene_indices)
