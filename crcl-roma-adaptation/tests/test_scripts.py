from pathlib import Path


def test_roma_launchers_cover_all_datasets_and_encoders():
    train_script = Path("scripts/train_roma.sh").read_text(encoding="utf-8")
    matrix_script = Path("scripts/run_roma_matrix.sh").read_text(encoding="utf-8")

    assert "train_roma.py" in train_script
    assert "--module_name SGR" in train_script
    assert "--img_dim 1024" in train_script
    assert "scenedepict,scanrefer,nr3d,3dllm" in matrix_script
    assert "bigru,bert" in matrix_script
