import json
from types import SimpleNamespace

import numpy as np

from data import get_precomp_loader


class _Vocab:
    word2idx = {"<mask>": 3}

    def __call__(self, token):
        return {"<start>": 1, "<end>": 2}.get(token, 4)

    def __len__(self):
        return 8


def test_roma_train_loader_uses_scene_unique_sampler_and_1024_patch_features(tmp_path):
    rows = [
        {"scene_id": "a", "description": "first chair"},
        {"scene_id": "a", "description": "second chair"},
        {"scene_id": "b", "description": "table"},
    ]
    (tmp_path / "scanrefer_train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    np.save(tmp_path / "pt2vec_200_random_train.npy", np.zeros((2, 200, 1024), dtype=np.float32))
    opt = SimpleNamespace(data_name="scanrefer", data_root=str(tmp_path), module_name="SGR", noise_ratio=0.0, text_enc_type="bigru", seed=5)

    loader = get_precomp_loader(str(tmp_path), "train", _Vocab(), opt, batch_size=2, shuffle=False, num_workers=0)
    images, image_lengths, _, _, ids = next(iter(loader))

    assert images.shape == (2, 200, 1024)
    assert image_lengths.tolist() == [200.0, 200.0]
    assert len({loader.dataset.scene_indices[index] for index in ids}) == len(ids)
