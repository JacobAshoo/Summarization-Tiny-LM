"""
Inference Script for Seq2Seq Summarization Transformer
=======================================================
Supports greedy decoding and beam search.

Usage:
    # Single string
    python predict.py --checkpoint ./checkpoints/ckpt_step0100000.pt \
                      --tokenizer  /home/jacob/release/bpe_tokenizer.json \
                      --text       "Your article text here..."

    # From a .txt file
    python predict.py --checkpoint ./checkpoints/ckpt_step0100000.pt \
                      --tokenizer  /home/jacob/release/bpe_tokenizer.json \
                      --file       article.txt

    # Beam search (default beam=1 i.e. greedy)
    python predict.py --checkpoint ... --tokenizer ... --text "..." --beam 5

    # Read from stdin
    echo "Article text..." | python predict.py --checkpoint ... --tokenizer ...
"""

import argparse
import sys
import math
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.amp import autocast

from model import (
    Seq2SeqTransformer,
    MAX_SRC_LEN, MAX_TGT_LEN,
)

# ── Special token IDs (must match train.py) ────────────────────────────────────
BOS_ID = 2
EOS_ID = 3
PAD_ID = 1


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> Seq2SeqTransformer:
    """
    Load model weights from a training checkpoint.
    Checkpoints are saved by train.py as:
        { "model": state_dict, "global_step": ..., "train_loss": ..., ... }
    gradient_checkpointing=False at inference — no backward pass, no need.
    """
    model = Seq2SeqTransformer(gradient_checkpointing=False).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model"]

    # Checkpoints are saved from a DDP-wrapped, torch.compiled model.
    # torch.compile adds a "_orig_mod." prefix; DDP adds "module.".
    # Strip whichever prefix is present so weights load cleanly.
    for prefix in ("_orig_mod.", "module."):
        if any(k.startswith(prefix) for k in state_dict):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    step = ckpt.get("global_step", "?")
    vl   = ckpt.get("val_loss",    "?")
    print(f"Loaded checkpoint  step={step}  val_loss={vl}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# TOKENIZATION
# ──────────────────────────────────────────────────────────────────────────────

def encode_source(text: str, tokenizer: Tokenizer, device: torch.device) -> torch.Tensor:
    """
    Tokenize source text, add BOS/EOS, truncate to MAX_SRC_LEN, return (1, S) tensor.
    Truncation mirrors the max_text=2000 filter in preprocessing (plus 2 special tokens).
    """
    ids = tokenizer.encode(text).ids
    ids = ids[: MAX_SRC_LEN - 2]           # leave room for BOS + EOS
    ids = [BOS_ID] + ids + [EOS_ID]
    return torch.tensor([ids], dtype=torch.long, device=device)  # (1, S)


# ──────────────────────────────────────────────────────────────────────────────
# GREEDY DECODING
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(
    model:   Seq2SeqTransformer,
    src:     torch.Tensor,          # (1, S)
    device:  torch.device,
    max_len: int = MAX_TGT_LEN,
) -> list[int]:
    """
    Autoregressive greedy decoding — always picks the highest-probability token.
    Fast but can be repetitive; use beam search for better quality.
    """
    with autocast(device_type=device.type, dtype=torch.bfloat16):
        enc_out = model.encode(src)                         # (1, S, D)

    tgt = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)  # (1, 1)

    for _ in range(max_len):
        with autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model.decode_step(tgt, enc_out)        # (1, T, V)

        next_id = logits[:, -1, :].argmax(dim=-1).item()   # greedy pick
        if next_id == EOS_ID:
            break
        tgt = torch.cat(
            [tgt, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )

    return tgt[0, 1:].tolist()   # strip leading BOS


# ──────────────────────────────────────────────────────────────────────────────
# BEAM SEARCH
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def beam_search(
    model:       Seq2SeqTransformer,
    src:         torch.Tensor,      # (1, S)
    device:      torch.device,
    beam_size:   int = 5,
    max_len:     int = MAX_TGT_LEN,
    length_pen:  float = 0.6,       # α in ((5+|Y|)/(5+1))^α — Google NMT style
) -> list[int]:
    """
    Beam search decoding with length penalty.

    Maintains `beam_size` hypotheses at each step, scoring them by cumulative
    log-probability normalized by a length penalty so the model doesn't always
    prefer short summaries.

    length_pen=0.6 is the Google NMT default; increase toward 1.0 to favour
    longer outputs, decrease toward 0.0 to favour shorter ones.
    """
    with autocast(device_type=device.type, dtype=torch.bfloat16):
        enc_out = model.encode(src)                         # (1, S, D)

    # Expand encoder output to (beam_size, S, D) for batched decoding
    enc_out = enc_out.expand(beam_size, -1, -1)

    # Each beam: (token_ids, cumulative_log_prob)
    beams: list[tuple[list[int], float]] = [([BOS_ID], 0.0)]
    completed: list[tuple[list[int], float]] = []

    for _ in range(max_len):
        if not beams:
            break

        # Build decoder input from all active beams — (num_beams, T)
        num_beams = len(beams)
        tgt = torch.tensor(
            [b[0] for b in beams], dtype=torch.long, device=device
        )

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model.decode_step(tgt, enc_out[:num_beams])  # (B, T, V)

        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)       # (B, V)

        # Expand each beam into top-k candidates
        candidates: list[tuple[list[int], float]] = []
        for i, (tokens, score) in enumerate(beams):
            top_lp, top_ids = log_probs[i].topk(beam_size)
            for lp, tok in zip(top_lp.tolist(), top_ids.tolist()):
                candidates.append((tokens + [tok], score + lp))

        # Sort by score, keep top beam_size open beams; siphon off completed
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = []
        for tokens, score in candidates:
            if tokens[-1] == EOS_ID:
                completed.append((tokens[1:-1], score))  # strip BOS + EOS
            else:
                beams.append((tokens, score))
            if len(beams) == beam_size:
                break

    # If nothing completed (hit max_len), take the best open beam
    if not completed:
        completed = [(b[0][1:], b[1]) for b in beams]  # strip BOS

    def length_penalty(seq_len: int) -> float:
        return ((5 + seq_len) / 6) ** length_pen

    best_tokens, best_score = max(
        completed,
        key=lambda x: x[1] / length_penalty(len(x[0])),
    )
    return best_tokens


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize text with a trained checkpoint.")
    p.add_argument("--checkpoint", required=True,  help="Path to .pt checkpoint file")
    p.add_argument("--tokenizer",  required=True,  help="Path to bpe_tokenizer.json")
    p.add_argument("--text",       default=None,   help="Article text to summarize")
    p.add_argument("--file",       default=None,   help="Path to a .txt file to summarize")
    p.add_argument("--beam",       type=int, default=1,
                   help="Beam size (1 = greedy, ≥2 = beam search)")
    p.add_argument("--length-pen", type=float, default=0.6,
                   help="Length penalty α for beam search (default 0.6)")
    p.add_argument("--max-len",    type=int, default=MAX_TGT_LEN,
                   help=f"Max tokens to generate (default {MAX_TGT_LEN})")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve input text ─────────────────────────────────────────────────────
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("Provide --text, --file, or pipe text via stdin.", file=sys.stderr)
        sys.exit(1)

    text = text.strip()
    if not text:
        print("Input text is empty.", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load tokenizer and model ───────────────────────────────────────────────
    tokenizer = Tokenizer.from_file(args.tokenizer)
    model     = load_model(args.checkpoint, device)

    # ── Encode source ──────────────────────────────────────────────────────────
    src = encode_source(text, tokenizer, device)
    print(f"Source tokens: {src.shape[1]}")

    # ── Decode ────────────────────────────────────────────────────────────────
    if args.beam <= 1:
        print("Decoding: greedy")
        token_ids = greedy_decode(model, src, device, max_len=args.max_len)
    else:
        print(f"Decoding: beam search  beam={args.beam}  length_pen={args.length_pen}")
        token_ids = beam_search(
            model, src, device,
            beam_size=args.beam,
            max_len=args.max_len,
            length_pen=args.length_pen,
        )

    # ── Decode token IDs back to text ──────────────────────────────────────────
    summary = tokenizer.decode(token_ids, skip_special_tokens=True)

    print(f"\n{'─' * 60}")
    print("SUMMARY:")
    print(f"{'─' * 60}")
    print(summary)
    print(f"{'─' * 60}")
    print(f"Output tokens: {len(token_ids)}")


if __name__ == "__main__":
    main()
