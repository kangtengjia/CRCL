#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import data
from model.CRCL import CRCL
from roma_config import require_bert_path
from roma_runs import format_markdown_metrics
from train_roma import load_vocab, validate


def evaluate(model_path: str, data_root: str, output_json: str | None = None) -> dict:
    checkpoint = torch.load(model_path, map_location="cpu")
    opt = checkpoint["opt"]
    opt.data_root = data_root
    opt.data_path = data_root
    opt.workers = int(getattr(opt, "workers", 0))
    if opt.text_enc_type == "bert":
        opt.bert_path = str(require_bert_path(opt.bert_path))
    vocab = load_vocab(opt)
    loader = data.get_precomp_loader(data_root, "val", vocab, opt, opt.batch_size, False, opt.workers)
    model = CRCL(opt)
    model.load_state_dict(checkpoint["model"])
    metrics = validate(opt, loader, model)
    result = {
        "dataset": opt.data_name,
        "text_encoder": opt.text_enc_type,
        "checkpoint": str(Path(model_path).resolve()),
        "queries": len(loader.dataset),
        "scenes": len(loader.dataset.feature_scene_ids),
        "metrics": metrics,
    }
    if output_json:
        target = Path(output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        target.with_suffix(".md").write_text(format_markdown_metrics(opt.data_name, opt.text_enc_type, metrics), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CRCL on a RoMa text-to-3D dataset.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_json")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model_path, args.data_root, args.output_json), indent=2))


if __name__ == "__main__":
    main()
