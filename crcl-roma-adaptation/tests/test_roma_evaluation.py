import numpy as np

from roma_evaluation import text_to_scene_metrics


def test_text_to_scene_metrics_supports_variable_captions_per_scene():
    similarities = np.asarray([
        [0.9, 0.1, 0.0],
        [0.7, 0.8, 0.1],
        [0.1, 0.9, 0.2],
        [0.3, 0.4, 0.8],
    ])

    metrics = text_to_scene_metrics(similarities, [0, 0, 1, 2], ks=(1, 2, 3))

    assert metrics["R@1"] == 75.0
    assert metrics["R@2"] == 100.0
    assert metrics["Rsum"] == 275.0
    assert metrics["MedR"] == 1.0
