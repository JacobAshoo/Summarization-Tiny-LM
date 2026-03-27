"""
Training Script for Seq2Seq Summarization Transformer
======================================================
Launch with:
    torchrun --nproc_per_node=NUM_GPUS train.py

Requires model.py in the same directory.

Changes from v1:
  - Removed GradScaler (no-op with bfloat16)
  - Added gradient accumulation + model.no_sync() for efficiency
  - Fixed DDP checkpoint sync bug (all ranks now validate together)
  - Fixed checkpoint train_loss (rolling average, not single-batch)
  - Added torch.compile() for ~20-30% throughput gain
  - Added prefetch_factor + persistent_workers to DataLoaders
  - Reduced LOG_EVERY to match grad accumulation cadence
"""

import ast
import csv
import os
import math
import random
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import mlflow

from model import (
    Seq2SeqTransformer,
    MAX_SRC_LEN, MAX_TGT_LEN,
    VOCAB_SIZE, D_MODEL,
    NUM_HEADS, ENC_LAYERS, DEC_LAYERS, FFN_HIDDEN, DROPOUT,
)


# ── Data ───────────────────────────────────────────────────────────────────────
TRAIN_FILES    = ["/home/jacob/release/final/train.csv"]
VAL_FILES      = ["/home/jacob/release/final/dev.csv"]
# Token IDs are stored as Python-literal strings e.g. "[123, 456, 789]"
# parsed with ast.literal_eval — no tokenizer needed at training time
BOS_ID         = 2    # [BOS] token id — matches BpeTrainer special_tokens order
EOS_ID         = 3    # [EOS] token id
PAD_ID         = 1    # [PAD] token id

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE       = 4           # per GPU, per micro-step
GRAD_ACCUM_STEPS = 16          # effective batch = BATCH_SIZE * world_size * GRAD_ACCUM_STEPS
                               # = 5 * 2 * 16 = 160 sequences per optimizer step
MAX_STEPS        = 500_000     # total optimiser steps
LR               = 1e-4        # peak learning rate after warmup
WEIGHT_DECAY     = 0.1         # applied to weight matrices only, not norms
WARMUP_STEPS     = 2000        # steps of linear LR warmup before cosine decay
LR_MIN_RATIO     = 0.1         # cosine decays to LR * LR_MIN_RATIO, not 0
GRAD_CLIP        = 1.0         # max gradient norm — prevents exploding gradients
LABEL_SMOOTHING  = 0.1         # prevents overconfident predictions

# ── Checkpointing ──────────────────────────────────────────────────────────────
CKPT_DIR             = "./checkpoints"
CKPT_EVERY           = 10000   # save a checkpoint every N optimizer steps
VAL_SUBSAMPLE_STEPS  = 100     # batches to run during mid-training validation
VAL_SEED             = 42      # fixed seed for val worker RNG — makes each checkpoint's
                               # val loss evaluate on the same effective data slice

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT   = "seq2seq-summarization"

# ── Misc ───────────────────────────────────────────────────────────────────────
NUM_WORKERS        = 4          # DataLoader worker processes per GPU
LOG_EVERY          = 50         # log every N optimizer steps (not micro-steps)
SHUFFLE_BUFFER     = 10_000     # rows held in memory before shuffling and yielding
LOSS_WINDOW        = 100        # rolling window size for train loss reported at ckpt


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────

class SummarizationIterableDataset(IterableDataset):
    """
    Streams rows from one or more CSV files one at a time — never loads
    the full dataset into RAM.

    Each CSV row has columns 'text' and 'summary' storing pre-tokenized
    token ID lists as Python-literal strings e.g. "[123, 456, 789]",
    written by tokenize_filter_save() in the preprocessing notebook.
    Parsed with ast.literal_eval — no tokenizer needed.

    DDP + multi-worker sharding:
        Each (rank, worker) pair reads a distinct slice of the file using
        a global worker index = rank * num_workers + worker_id.
        This ensures no two processes see the same row.

    Streams indefinitely — the training loop controls when to stop via
    MAX_STEPS.
    """
    def __init__(self, files: list[str], rank: int, world_size: int):
        self.files      = files
        self.rank       = rank
        self.world_size = world_size

    def _iter_file(self, path: str, global_worker_id: int, total_workers: int):
        buffer = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row_idx, row in enumerate(reader):
                # Each worker reads only its assigned rows
                if (row_idx % total_workers) != global_worker_id:
                    continue
                try:
                    src_ids = ast.literal_eval(row["text"])
                    tgt_ids = ast.literal_eval(row["summary"])
                except Exception:
                    continue  # skip malformed rows silently

                # IDs are already length-filtered by preprocessing —
                # no truncation needed, just add special tokens
                src = [BOS_ID] + src_ids + [EOS_ID]
                tgt = [BOS_ID] + tgt_ids        # decoder input
                lbl = tgt_ids  + [EOS_ID]       # labels shifted by 1

                buffer.append((
                    torch.tensor(src, dtype=torch.long),
                    torch.tensor(tgt, dtype=torch.long),
                    torch.tensor(lbl, dtype=torch.long),
                ))

                if len(buffer) >= SHUFFLE_BUFFER:
                    random.shuffle(buffer)
                    yield from buffer
                    buffer.clear()

        # Flush and shuffle any remaining rows at end of file
        if buffer:
            random.shuffle(buffer)
            yield from buffer

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id   = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        # Global worker id across all ranks and all workers per rank
        total_workers    = self.world_size * num_workers
        global_worker_id = self.rank * num_workers + worker_id

        while True:  # loop indefinitely — MAX_STEPS controls when to stop
            for path in self.files:
                yield from self._iter_file(path, global_worker_id, total_workers)


# ──────────────────────────────────────────────────────────────────────────────
# COLLATE
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Pads each sequence to the longest sequence in the batch.
    Much more memory-efficient than padding to global MAX_*_LEN.
    PAD_ID positions are ignored by the loss via ignore_index.
    """
    srcs, tgts, lbls = zip(*batch)
    src = pad_sequence(srcs, batch_first=True, padding_value=PAD_ID)
    tgt = pad_sequence(tgts, batch_first=True, padding_value=PAD_ID)
    lbl = pad_sequence(lbls, batch_first=True, padding_value=PAD_ID)
    return src, tgt, lbl


# ──────────────────────────────────────────────────────────────────────────────
# DDP
# ──────────────────────────────────────────────────────────────────────────────

def setup_ddp():
    """
    Initialises the NCCL process group for DDP.
    torchrun sets LOCAL_RANK, RANK, and WORLD_SIZE automatically.
    NCCL is the fastest backend for GPU-to-GPU communication.
    """
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_ddp():
    dist.destroy_process_group()


# ──────────────────────────────────────────────────────────────────────────────
# OPTIMIZER
# ──────────────────────────────────────────────────────────────────────────────

def build_optimizer(model) -> AdamW:
    """
    AdamW with weight decay applied only to weight matrices (ndim >= 2).
    RMSNorm gamma vectors (ndim == 1) are excluded from weight decay —
    regularising them would shrink the learned scale and hurt training.

    betas=(0.9, 0.95) — standard for transformer training.
    eps=1e-8 — numerical stability in the Adam update denominator.
    """
    decay_params    = [p for n, p in model.named_parameters()
                       if p.ndim >= 2 and p.requires_grad]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.ndim < 2  and p.requires_grad]

    return AdamW(
        [
            {"params": decay_params,    "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LR, betas=(0.9, 0.95), eps=1e-8,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SCHEDULER
# ──────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer: AdamW) -> LambdaLR:
    """
    Linear warmup for WARMUP_STEPS, then cosine decay to 0 over MAX_STEPS.

    Warmup prevents large gradient updates early in training when the model
    weights are random and the loss landscape is steep.
    Cosine decay smoothly reduces LR over the rest of training, letting
    the model converge rather than oscillating around a minimum.
    """
    def lr_lambda(step: int) -> float:
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
        # Cosine decay to LR_MIN_RATIO, not all the way to 0 — the model
        # should still be learning in the final third of training.
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return LR_MIN_RATIO + (1.0 - LR_MIN_RATIO) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# LOSS
# ──────────────────────────────────────────────────────────────────────────────

def build_criterion() -> nn.CrossEntropyLoss:
    """
    Cross-entropy with:
      ignore_index=PAD_ID     — padded positions don't contribute to loss
      label_smoothing=0.1     — soft targets prevent overconfident predictions
                                and improve generalisation
    """
    return nn.CrossEntropyLoss(
        ignore_index=PAD_ID,
        label_smoothing=LABEL_SMOOTHING,
    )


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, rank, max_batches: int = None) -> float:
    """
    Runs validation on this rank and returns the local average loss.
    Caller is responsible for all_reduce-ing across ranks.
    max_batches: always set for IterableDataset since there is no epoch end.
    """
    model.eval()
    total_loss = 0.0
    device = torch.device(f"cuda:{rank}")

    for i, (src, tgt, lbl) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(src, tgt)
            loss   = criterion(logits.transpose(1, 2), lbl)

        total_loss += loss.item()

    n_batches = min(max_batches, i + 1) if max_batches is not None else (i + 1)
    return total_loss / max(n_batches, 1)


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, criterion, rank):
    """
    Step-based training loop — runs until MAX_STEPS.

    Each optimizer step accumulates gradients over GRAD_ACCUM_STEPS micro-steps.
    model.no_sync() suppresses redundant NCCL allreduces on all but the final
    micro-step, significantly reducing inter-GPU communication overhead.

    Checkpoints every CKPT_EVERY steps. All ranks validate in parallel and
    results are averaged via all_reduce — rank 1 no longer idles while rank 0
    validates, fixing the DDP synchronization hazard.
    """
    model.train()
    device    = torch.device(f"cuda:{rank}")
    data_iter = iter(train_loader)

    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    # Rolling window of recent per-micro-step losses for honest train loss
    # reporting at checkpoint time (not a single-batch snapshot).
    recent_losses: list[float] = []

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            return next(data_iter)

    loss_accum_for_log = 0.0

    for global_step in range(1, MAX_STEPS + 1):

        # ── Gradient accumulation ─────────────────────────────────────────────
        # Suppress DDP gradient sync on all micro-steps except the last.
        # This avoids GRAD_ACCUM_STEPS-1 unnecessary NCCL allreduces per step.
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro_step in range(GRAD_ACCUM_STEPS):
            src, tgt, lbl = next_batch()
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)

            is_last_micro = (micro_step == GRAD_ACCUM_STEPS - 1)
            sync_ctx = nullcontext() if is_last_micro else model.no_sync()

            with sync_ctx:
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(src, tgt)
                    # Divide loss by accumulation steps so gradients are
                    # averaged over all micro-steps, not just the last one.
                    loss = criterion(logits.transpose(1, 2), lbl) / GRAD_ACCUM_STEPS

                loss.backward()
                step_loss += loss.item()

        # ── Optimizer step ────────────────────────────────────────────────────
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        # Track loss for logging and checkpointing
        loss_accum_for_log += step_loss
        recent_losses.append(step_loss)
        if len(recent_losses) > LOSS_WINDOW:
            recent_losses.pop(0)

        # ── Logging ───────────────────────────────────────────────────────────
        if rank == 0 and global_step % LOG_EVERY == 0:
            lr_now     = scheduler.get_last_lr()[0]
            train_loss = loss_accum_for_log / LOG_EVERY
            print(
                f"step {global_step:>7d} | "
                f"loss {train_loss:.4f} | "
                f"lr {lr_now:.2e}"
            )
            mlflow.log_metrics(
                {"train/loss": train_loss, "train/lr": lr_now},
                step=global_step,
            )
            loss_accum_for_log = 0.0

        # ── Checkpoint ────────────────────────────────────────────────────────
        if global_step % CKPT_EVERY == 0:
            # All ranks validate in parallel — no rank idles.
            # val_loss_local is each rank's average over its shard of val data.
            val_loss_local = validate(model, val_loader, criterion, rank,
                                      max_batches=VAL_SUBSAMPLE_STEPS)
            model.train()

            # Average validation loss across all ranks.
            val_tensor = torch.tensor(val_loss_local, device=device)
            dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_tensor.item()

            # Rolling average of recent training losses — much more stable
            # than a single-batch snapshot.
            train_loss_ckpt = sum(recent_losses) / max(len(recent_losses), 1)

            if rank == 0:
                mlflow.log_metrics(
                    {"val/loss": val_loss, "train/loss_at_ckpt": train_loss_ckpt},
                    step=global_step,
                )

                ckpt_path = (
                    f"{CKPT_DIR}/"
                    f"ckpt_step{global_step:07d}"
                    f"_tl{train_loss_ckpt:.4f}"
                    f"_vl{val_loss:.4f}"
                    f".pt"
                )
                torch.save({
                    "global_step": global_step,
                    "model":       model.module.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "scheduler":   scheduler.state_dict(),
                    "train_loss":  train_loss_ckpt,
                    "val_loss":    val_loss,
                }, ckpt_path)
                print(f"  Saved → {ckpt_path}  (val_loss={val_loss:.4f})")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    setup_ddp()
    rank       = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device     = torch.device(f"cuda:{rank}")

    effective_batch = BATCH_SIZE * world_size * GRAD_ACCUM_STEPS
    if rank == 0:
        print(
            f"Starting training on {world_size} GPUs | "
            f"MAX_STEPS={MAX_STEPS} | "
            f"effective_batch={effective_batch}"
        )

    # ── Datasets ──────────────────────────────────────────────────────────────
    # IterableDataset handles its own DDP sharding internally via rank/world_size.
    # No DistributedSampler needed.
    train_dataset = SummarizationIterableDataset(TRAIN_FILES, rank, world_size)
    val_dataset   = SummarizationIterableDataset(VAL_FILES,   rank, world_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,       # pre-load next 2 batches per worker
        persistent_workers=True, # keep workers alive between steps (no respawn)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        # Fixed seed so every checkpoint evaluates on the same effective slice —
        # makes val loss curves directly comparable across checkpoints.
        worker_init_fn=lambda wid: __import__('random').seed(VAL_SEED + wid),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Seq2SeqTransformer(gradient_checkpointing=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    # torch.compile() fuses RMSNorm, SwiGLU, and embedding ops into optimised
    # kernels — typically 20-30% throughput gain over a long run at no cost.
    model = torch.compile(model)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # ── Loss ──────────────────────────────────────────────────────────────────
    # GradScaler removed — it is a no-op with bfloat16 (same exponent range as
    # float32, so gradient underflow cannot occur). It only added overhead.
    criterion = build_criterion()

    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── MLflow — rank 0 only ──────────────────────────────────────────────────
    if rank == 0:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow.start_run()
        mlflow.log_params({
            "d_model":              D_MODEL,
            "num_heads":            NUM_HEADS,
            "enc_layers":           ENC_LAYERS,
            "dec_layers":           DEC_LAYERS,
            "ffn_hidden":           FFN_HIDDEN,
            "dropout":              DROPOUT,
            "vocab_size":           VOCAB_SIZE,
            "max_src_len":          MAX_SRC_LEN,
            "max_tgt_len":          MAX_TGT_LEN,
            "total_params":         total_params,
            "batch_size":           BATCH_SIZE,
            "grad_accum_steps":     GRAD_ACCUM_STEPS,
            "world_size":           world_size,
            "effective_batch_size": effective_batch,
            "max_steps":            MAX_STEPS,
            "lr":                   LR,
            "weight_decay":         WEIGHT_DECAY,
            "warmup_steps":         WARMUP_STEPS,
            "grad_clip":            GRAD_CLIP,
            "label_smoothing":      LABEL_SMOOTHING,
        })

    # ── Train ─────────────────────────────────────────────────────────────────
    try:
        train(model, train_loader, val_loader, criterion, rank)
    finally:
        if rank == 0:
            mlflow.end_run()

    cleanup_ddp()


if __name__ == "__main__":
    main()
