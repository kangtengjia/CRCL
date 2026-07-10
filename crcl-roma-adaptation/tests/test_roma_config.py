import pytest

from roma_config import require_bert_path, vocab_filename


def test_bert_path_requires_local_model_files(tmp_path):
    with pytest.raises(ValueError, match="BERT_PATH"):
        require_bert_path("")
    model_path = tmp_path / "bert"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}")
    (model_path / "vocab.txt").write_text("[PAD]\n")
    (model_path / "model.safetensors").write_bytes(b"weights")

    assert require_bert_path(model_path) == model_path
    assert vocab_filename("nr3d") == "nr3d_vocab.json"
    assert vocab_filename("scanrefer") == "my_data_vocab.json"
