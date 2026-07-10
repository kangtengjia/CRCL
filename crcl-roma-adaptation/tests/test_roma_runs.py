from roma_runs import RUN_PRESETS, stage_epoch_totals


def test_roma_presets_cover_eight_runs_with_four_stage_totals():
    assert set(RUN_PRESETS) == {
        ("scenedepict", "bigru"),
        ("scenedepict", "bert"),
        ("scanrefer", "bigru"),
        ("scanrefer", "bert"),
        ("nr3d", "bigru"),
        ("nr3d", "bert"),
        ("3dllm", "bigru"),
        ("3dllm", "bert"),
    }
    assert stage_epoch_totals(300) == (40, 40, 40, 180)
    assert stage_epoch_totals(30) == (4, 4, 4, 18)
    assert RUN_PRESETS[("nr3d", "bigru")].batch_size == 8
