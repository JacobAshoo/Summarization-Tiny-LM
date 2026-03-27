"""
Lead Bias Analysis for Summarization Datasets
==============================================
Measures how much of each reference summary comes from the beginning
vs. the middle/end of its source document.

Works with the cleaned CSVs that have raw 'text' and 'summary' columns
(i.e. the files BEFORE tokenization — the ones in cleaned/).

Usage:
    python analyze_lead_bias.py /home/jacob/release/cleaned/train_0.csv [...]

Pass one or more CSV files. The script samples up to --max-samples rows
(default 50,000) across all files for speed.
"""

import argparse
import csv
import re
import sys
import json
from collections import Counter, defaultdict
from pathlib import Path

# ── Sentence splitting ────────────────────────────────────────────────────────

_SENT_RE = re.compile(
    r'(?<=[.!?])'       # lookbehind: sentence-ending punctuation
    r'(?:\s+)'          # whitespace between sentences
    r'(?=[A-Z"\'])'     # lookahead: next sentence starts with upper/quote
)

def split_sentences(text: str) -> list[str]:
    """Rough but fast sentence splitter — good enough for positional analysis."""
    sents = _SENT_RE.split(text.strip())
    return [s.strip() for s in sents if s.strip()]


# ── N-gram utilities ──────────────────────────────────────────────────────────

def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenization — sufficient for overlap analysis."""
    return text.lower().split()


# ── Core analysis ─────────────────────────────────────────────────────────────

def compute_positional_overlap(
    source: str,
    summary: str,
    n: int = 3,
    num_buckets: int = 10,
) -> dict:
    """
    Splits the source into sentences, groups them into positional buckets,
    and computes what fraction of summary n-grams appear in each bucket.

    Returns:
        {
            "bucket_overlaps": [float, ...],   # fraction of summary n-grams found per bucket
            "num_source_sents": int,
            "num_summary_ngrams": int,
            "any_overlap": bool,
        }
    """
    source_sents = split_sentences(source)
    if len(source_sents) < 2:
        return None  # skip trivially short documents

    summary_toks = tokenize(summary)
    summary_ngs = set(ngrams(summary_toks, n))
    if not summary_ngs:
        return None

    # Assign each sentence to a positional bucket (0 = start, num_buckets-1 = end)
    bucket_ngrams = defaultdict(set)
    for i, sent in enumerate(source_sents):
        bucket = min(int(i / len(source_sents) * num_buckets), num_buckets - 1)
        sent_toks = tokenize(sent)
        for ng in ngrams(sent_toks, n):
            bucket_ngrams[bucket].add(ng)

    # What fraction of summary n-grams appear in each bucket?
    bucket_overlaps = []
    for b in range(num_buckets):
        overlap = len(summary_ngs & bucket_ngrams[b])
        bucket_overlaps.append(overlap / len(summary_ngs))

    return {
        "bucket_overlaps": bucket_overlaps,
        "num_source_sents": len(source_sents),
        "num_summary_ngrams": len(summary_ngs),
        "any_overlap": any(o > 0 for o in bucket_overlaps),
    }


def compute_lead_n_rouge(
    source: str,
    summary: str,
    lead_n: int = 3,
    ngram_size: int = 1,
) -> float:
    """
    ROUGE-like unigram overlap between the first `lead_n` source sentences
    and the full summary. Returns recall (fraction of summary unigrams
    found in the lead). High values = strong lead bias.
    """
    source_sents = split_sentences(source)
    lead_text = " ".join(source_sents[:lead_n])

    lead_toks = Counter(tokenize(lead_text))
    summary_toks = Counter(tokenize(summary))

    if not summary_toks:
        return 0.0

    overlap = sum((lead_toks & summary_toks).values())
    return overlap / sum(summary_toks.values())


# ── File loading ──────────────────────────────────────────────────────────────

def iter_rows(paths: list[str], max_samples: int):
    """Yield (text, summary) pairs from one or more CSVs, up to max_samples."""
    count = 0
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if count >= max_samples:
                    return
                text = row.get("text", "")
                summary = row.get("summary", "")
                if text and summary:
                    yield text, summary
                    count += 1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze lead bias in a summarization dataset.")
    parser.add_argument("files", nargs="+", help="CSV files with 'text' and 'summary' columns")
    parser.add_argument("--max-samples", type=int, default=50_000,
                        help="Max rows to analyze (default: 50,000)")
    parser.add_argument("--ngram", type=int, default=3,
                        help="N-gram size for positional overlap (default: 3)")
    parser.add_argument("--buckets", type=int, default=10,
                        help="Number of positional buckets (default: 10)")
    parser.add_argument("--lead-n", type=int, default=3,
                        help="Number of lead sentences for lead-N baseline (default: 3)")
    parser.add_argument("--json", action="store_true",
                        help="Also dump results as JSON to stdout")
    args = parser.parse_args()

    print(f"Analyzing lead bias  (ngram={args.ngram}, buckets={args.buckets}, "
          f"lead_n={args.lead_n}, max_samples={args.max_samples:,})")
    print(f"Files: {args.files}\n")

    # Accumulate per-bucket overlap across all examples
    bucket_sums = [0.0] * args.buckets
    lead_recalls = []
    skipped = 0
    analyzed = 0

    for text, summary in iter_rows(args.files, args.max_samples):
        result = compute_positional_overlap(text, summary, n=args.ngram,
                                            num_buckets=args.buckets)
        if result is None:
            skipped += 1
            continue

        for b, val in enumerate(result["bucket_overlaps"]):
            bucket_sums[b] += val

        lead_recalls.append(
            compute_lead_n_rouge(text, summary, lead_n=args.lead_n)
        )
        analyzed += 1

        if analyzed % 5000 == 0:
            print(f"  ...processed {analyzed:,} rows")

    if analyzed == 0:
        print("ERROR: no valid rows found. Check that your CSVs have 'text' and 'summary' columns.")
        sys.exit(1)

    # ── Results ───────────────────────────────────────────────────────────────
    bucket_avgs = [s / analyzed for s in bucket_sums]
    avg_lead_recall = sum(lead_recalls) / len(lead_recalls)

    # What fraction of total overlap comes from each bucket?
    total_overlap = sum(bucket_avgs)
    bucket_pcts = [(b / total_overlap * 100) if total_overlap > 0 else 0 for b in bucket_avgs]

    # How much overlap in first 20% vs last 80%?
    top_20_cutoff = max(1, args.buckets // 5)
    first_20_pct = sum(bucket_pcts[:top_20_cutoff])
    first_50_pct = sum(bucket_pcts[:args.buckets // 2])

    print(f"\n{'=' * 65}")
    print(f"  LEAD BIAS ANALYSIS — {analyzed:,} documents (skipped {skipped:,})")
    print(f"{'=' * 65}\n")

    # Bar chart
    max_bar = 40
    max_pct = max(bucket_pcts) if bucket_pcts else 1
    print(f"  {'Position':<14} {'Overlap %':>10}   Distribution")
    print(f"  {'─' * 14} {'─' * 10}   {'─' * max_bar}")
    for b in range(args.buckets):
        lo = b / args.buckets * 100
        hi = (b + 1) / args.buckets * 100
        label = f"{lo:.0f}–{hi:.0f}%"
        bar_len = int(bucket_pcts[b] / max_pct * max_bar) if max_pct > 0 else 0
        bar = "█" * bar_len
        print(f"  {label:<14} {bucket_pcts[b]:>9.1f}%   {bar}")

    print(f"\n  Summary n-gram overlap concentrated in...")
    print(f"    First 20% of source:  {first_20_pct:.1f}% of all overlap")
    print(f"    First 50% of source:  {first_50_pct:.1f}% of all overlap")

    print(f"\n  Lead-{args.lead_n} baseline (unigram recall):")
    print(f"    Mean:   {avg_lead_recall:.3f}")
    print(f"    Median: {sorted(lead_recalls)[len(lead_recalls) // 2]:.3f}")

    # Interpretation
    print(f"\n{'─' * 65}")
    if first_20_pct > 60:
        print("  ⚠  STRONG lead bias: >60% of summary content comes from the")
        print("     first 20% of the source. The model will likely learn to copy")
        print("     the opening sentences. Consider sentence shuffling during")
        print("     training or filtering for examples with distributed coverage.")
    elif first_20_pct > 40:
        print("  ⚡ MODERATE lead bias: 40–60% of overlap in the first 20%.")
        print("     Common in news datasets. Sentence shuffling augmentation")
        print("     during training would help.")
    else:
        print("  ✓  LOW lead bias: summary content is fairly distributed across")
        print("     the source. Lead bias is probably not your main issue.")
    print(f"{'─' * 65}")

    if args.json:
        print("\n" + json.dumps({
            "analyzed": analyzed,
            "skipped": skipped,
            "bucket_pcts": [round(p, 2) for p in bucket_pcts],
            "first_20_pct": round(first_20_pct, 2),
            "first_50_pct": round(first_50_pct, 2),
            "lead_recall_mean": round(avg_lead_recall, 4),
        }, indent=2))


if __name__ == "__main__":
    main()
