import pytest
import torch

from checkpointing import build_checkpoint, restore_checkpoint


def test_checkpoint_restores_optimizer_and_crcl_self_refining_state():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint = build_checkpoint(
        stage_index=2,
        stage_epoch=4,
        global_epoch=25,
        model_state=model.state_dict(),
        optimizer=optimizer,
        move_gt=[0.2, 0.8],
        best_score=123.0,
        best_metrics={"Rsum": 123.0},
        iterations=9,
        config={"data_name": "scanrefer"},
    )

    state = restore_checkpoint(checkpoint, optimizer)

    assert state.stage_index == 2
    assert state.global_epoch == 25
    assert state.move_gt == [0.2, 0.8]
    assert state.best_score == 123.0


def test_legacy_checkpoint_requires_explicit_weights_only_mode():
    optimizer = torch.optim.Adam(torch.nn.Linear(1, 1).parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="legacy"):
        restore_checkpoint({"epoch": 2, "model": []}, optimizer)
