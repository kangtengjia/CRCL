#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import tensorboard_logger as tb_logger
import torch
from transformers import BertTokenizer

import data
import opts
from checkpointing import build_checkpoint, restore_checkpoint
from evaluation import LogCollector, shard_attn_scores
from manifest import build_run_manifest
from model.CRCL import CRCL
from roma import canonical_dataset_name, source_paths
from roma_config import require_bert_path, vocab_filename
from roma_evaluation import DEFAULT_KS, text_to_scene_metrics
from roma_runs import stage_epoch_totals
from vocab import deserialize_vocab


ROMA_DATASETS = {"scenedepict", "scanrefer", "nr3d", "3dllm"}


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_vocab(opt):
    if opt.text_enc_type == "bert":
        opt.bert_path = str(require_bert_path(opt.bert_path))
        tokenizer = BertTokenizer.from_pretrained(opt.bert_path, local_files_only=True)
        opt.vocab_size = len(tokenizer.vocab)
        return tokenizer
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, vocab_filename(opt.data_name)))
    vocab.add_word("<mask>")
    opt.vocab_size = len(vocab)
    return vocab


def encode_scenes(model: CRCL, features: np.ndarray, batch_size: int) -> np.ndarray:
    encoded = []
    device = next(model.base_model.img_enc.parameters()).device
    for start in range(0, len(features), batch_size):
        batch = torch.as_tensor(features[start:start + batch_size], dtype=torch.float32, device=device)
        encoded.append(model.base_model.img_enc(batch).detach().cpu().numpy())
    return np.concatenate(encoded, axis=0)


def encode_captions(model: CRCL, loader):
    rows = [None] * len(loader.dataset)
    lengths = [0] * len(loader.dataset)
    for images, _, captions, caption_lengths, ids in loader:
        forward_result = model.forward_emb(images, None, captions, caption_lengths)
        if len(forward_result) == 3:
            _, cap_embs, cap_lens = forward_result
        elif len(forward_result) == 2:
            _, cap_embs = forward_result
            cap_lens = caption_lengths
        else:
            raise ValueError(f"unexpected forward_emb output size: {len(forward_result)}")
        cap_embs = cap_embs.detach().cpu().numpy()
        for row, index in enumerate(ids):
            rows[index] = cap_embs[row, : int(cap_lens[row])]
            lengths[index] = int(cap_lens[row])
    maximum = max(lengths, default=0)
    embedding_size = rows[0].shape[-1] if rows else 0
    result = np.zeros((len(rows), maximum, embedding_size), dtype=np.float32)
    for index, row in enumerate(rows):
        result[index, : lengths[index]] = row
    return result, lengths


def validate(opt, loader, model: CRCL) -> dict:
    model.val_start()
    with torch.no_grad():
        image_embs = encode_scenes(model, loader.dataset.images, opt.batch_size)
        caption_embs, caption_lengths = encode_captions(model, loader)
        similarities = shard_attn_scores(model.base_model, image_embs, caption_embs, caption_lengths, opt, shard_size=100)
    return text_to_scene_metrics(similarities.T, loader.dataset.scene_indices, ks=DEFAULT_KS)


def train_epoch(opt, loader, model: CRCL, local_epoch: int, stage_index: int, global_epoch: int) -> LogCollector:
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(global_epoch)
    collector = LogCollector()
    for images, image_lengths, captions, caption_lengths, text_ids in loader:
        scene_indices = [loader.dataset.scene_indices[index] for index in text_ids]
        if len(scene_indices) != len(set(scene_indices)):
            raise RuntimeError("RoMa sampler emitted duplicate scenes in one CRCL batch")
        model.train_start()
        model.logger = collector
        model.train_self(images, image_lengths, captions, caption_lengths, text_ids, epoch=local_epoch, schedule=stage_index)
    return collector


def main() -> None:
    opt = opts.parse_opt()
    opt.data_name = canonical_dataset_name(opt.data_name)
    if opt.data_name not in ROMA_DATASETS:
        raise ValueError("train_roma.py only supports RoMa datasets")
    if not opt.data_root:
        raise ValueError("--data_root is required")
    if opt.module_name != "SGR":
        raise ValueError("CRCL-RoMa uses the approved single-model SGR variant")
    opt.noise_ratio = 0.0
    opt.img_dim = 1024
    opt.embed_size = 1024
    opt.num_regions = 200
    set_seeds(opt.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    Path(opt.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    tb_logger.configure(opt.log_dir, flush_secs=5)
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)
    vocab = load_vocab(opt)
    train_loader = data.get_precomp_loader(opt.data_root, "train", vocab, opt, opt.batch_size, True, opt.workers)
    val_loader = data.get_precomp_loader(opt.data_root, "val", vocab, opt, opt.batch_size, False, opt.workers)
    totals = stage_epoch_totals(opt.num_epochs)
    if any(total <= opt.warm_epoch for total in totals):
        raise ValueError("each CRCL phase must exceed warm_epoch")
    manifest = build_run_manifest(source_paths(opt.data_root, opt.data_name, "train") + source_paths(opt.data_root, opt.data_name, "val"), repo_root=Path(__file__).parent)
    (Path(opt.checkpoint_dir) / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    best_score, best_metrics, global_epoch = 0.0, {}, 0
    resume = None
    if opt.resume:
        resume = torch.load(opt.resume, map_location="cpu")
    model = None
    move_gt = None
    for stage_index, stage_total in enumerate(totals):
        start_epoch = 0
        if resume and stage_index < int(resume.get("stage_index", 0)):
            global_epoch += stage_total
            continue
        model = CRCL(opt)
        model.move_gt = np.ones(train_loader.dataset.length, dtype=np.float32) if move_gt is None else move_gt
        if resume and stage_index == int(resume.get("stage_index", 0)):
            model.load_state_dict(resume["model"])
            state = restore_checkpoint(resume, model.optimizer, weights_only=opt.weights_only)
            model.move_gt = np.asarray(state.move_gt, dtype=np.float32)
            model.step = state.iterations
            start_epoch, global_epoch = state.stage_epoch, state.global_epoch
            best_score, best_metrics = state.best_score, state.best_metrics
        for local_epoch in range(start_epoch, stage_total):
            lr = opt.learning_rate * (0.1 ** (local_epoch // opt.lr_update))
            for group in model.optimizer.param_groups:
                group["lr"] = lr
            collector = train_epoch(opt, train_loader, model, local_epoch, stage_index, global_epoch)
            loss_meter = collector.meters.get("loss")
            mean_loss = float(loss_meter.avg) if loss_meter is not None else float("nan")
            group_lrs = [float(group["lr"]) for group in model.optimizer.param_groups]
            logger.info(
                "TRAIN_HEALTH stage=%s epoch=%s global=%s loss=%s lr=%s group_lrs=%s",
                stage_index,
                local_epoch,
                global_epoch,
                mean_loss,
                group_lrs[0] if group_lrs else None,
                group_lrs,
            )
            metrics = validate(opt, val_loader, model)
            score = metrics["Rsum"]
            if score > best_score:
                best_score, best_metrics = score, metrics
            for key, value in metrics.items():
                tb_logger.log_value(key.replace("@", ""), value, step=model.step)
            checkpoint = build_checkpoint(
                stage_index=stage_index, stage_epoch=local_epoch + 1, global_epoch=global_epoch + 1,
                model_state=model.state_dict(), optimizer=model.optimizer, move_gt=model.move_gt,
                best_score=best_score, best_metrics=best_metrics, iterations=model.step, config=vars(opt), input_manifest=manifest,
            )
            checkpoint["opt"] = opt
            torch.save(checkpoint, Path(opt.checkpoint_dir) / "checkpoint.pth.tar")
            if score >= best_score:
                torch.save(checkpoint, Path(opt.checkpoint_dir) / "model_best.pth.tar")
            logger.info(
                "stage=%s epoch=%s global=%s R@1=%.2f MRR=%.2f Rsum=%.2f best=%.2f",
                stage_index,
                local_epoch,
                global_epoch,
                metrics["R@1"],
                metrics["MRR"],
                score,
                best_score,
            )
            global_epoch += 1
        move_gt = model.move_gt
        resume = None
    logger.info("best CRCL-RoMa metrics: %s", best_metrics)


if __name__ == "__main__":
    main()
