import json

import numpy as np
import pytest

from roma import SceneUniqueBatchSampler, load_roma_bundle


def _json(path, rows):
    path.write_text(json.dumps(rows), encoding="utf-8")


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_scenedepict_maps_variable_captions_to_explicit_scene_indices(tmp_path):
    _json(tmp_path / "3D_Text_Retrv_train_final_sorted.json", [
        {"scene_id": "a", "description": "first"},
        {"scene_id": "a", "description": "second"},
        {"scene_id": "b", "description": "third"},
    ])
    np.save(tmp_path / "3D_Text_Retrv_grid_train.npy", np.zeros((2, 3, 4), dtype=np.float32))

    bundle = load_roma_bundle(tmp_path, "scenedepict", "train")

    assert bundle.scene_indices == [0, 0, 1]
    assert bundle.feature_scene_ids == ["a", "b"]
    assert bundle.features.shape == (2, 3, 4)


def test_nr3d_uses_scanrefer_then_scenedepict_feature_pools(tmp_path):
    _jsonl(tmp_path / "scanrefer_train.jsonl", [
        {"scene_id": "a", "description": "a"},
        {"scene_id": "b", "description": "b"},
    ])
    _jsonl(tmp_path / "scanrefer_val.jsonl", [{"scene_id": "c", "description": "c"}])
    _json(tmp_path / "3D_Text_Retrv_train_final_sorted.json", [{"scene_id": "fallback", "description": "f"}])
    _json(tmp_path / "3D_Text_Retrv_val_final_sorted.json", [])
    _jsonl(tmp_path / "nr3d_train.jsonl", [
        {"scan_id": "c", "description": "from scanrefer val"},
        {"scan_id": "fallback", "description": "from fallback"},
    ])
    np.save(tmp_path / "pt2vec_200_random_train.npy", np.zeros((2, 2, 4), dtype=np.float32))
    np.save(tmp_path / "pt2vec_200_random_val.npy", np.ones((1, 2, 4), dtype=np.float32))
    np.save(tmp_path / "3D_Text_Retrv_grid_train.npy", np.full((1, 2, 4), 2, dtype=np.float32))
    np.save(tmp_path / "3D_Text_Retrv_grid_val.npy", np.zeros((0, 2, 4), dtype=np.float32))

    bundle = load_roma_bundle(tmp_path, "nr3d", "train")

    assert bundle.feature_scene_ids == ["c", "fallback"]
    assert bundle.features[:, 0, 0].tolist() == [1.0, 2.0]


def test_scene_unique_batch_sampler_keeps_every_caption_and_rejects_large_batch():
    sampler = SceneUniqueBatchSampler([0, 0, 1, 2, 2], batch_size=3, shuffle=False)
    batches = list(sampler)

    assert sorted(index for batch in batches for index in batch) == list(range(5))
    assert all(len({sampler.scene_indices[index] for index in batch}) == len(batch) for batch in batches)
    with pytest.raises(ValueError, match="unique scenes"):
        SceneUniqueBatchSampler([0, 0, 1], batch_size=3)
