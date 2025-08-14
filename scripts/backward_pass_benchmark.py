#!/usr/bin/env python3
import os
import time
from datetime import datetime
from types import SimpleNamespace
from collections import defaultdict

import fire
import torch
import pandas as pd
import matplotlib.pyplot as plt

from mint.helpers.extract import load_config
from mint.utils.wrapper import ESMWrapper


def _device_ok(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use device='cpu' or enable GPU.")
    return torch.device(device)


def _make_tokens(batch_size: int, seq_len: int, chains: int, device: torch.device):
    toks, ids = [], []
    for c in range(chains):
        toks.append(torch.randint(1, 32, (batch_size, seq_len), dtype=torch.long))
        ids.append(torch.full((batch_size, seq_len), c, dtype=torch.long))
    tokens = torch.cat(toks, dim=1).to(device)
    chain_ids = torch.cat(ids, dim=1).to(device)
    return tokens, chain_ids


def _powers_of_two_up(cap: int, base: int = 16):
    """16, 32, 64, ... <= cap"""
    out = []
    b = base
    while b <= cap:
        out.append(b)
        b *= 2
    return out


def _powers_of_two_down(cap: int, start: int = 8):
    """min(start, cap), then halve: 8, 4, 2, 1 (filtered by <= cap)"""
    out = []
    b = min(start, cap)
    seen = set()
    while b >= 1:
        if b <= cap and b not in seen:
            out.append(b)
            seen.add(b)
        b //= 2
    if 1 not in seen and cap >= 1:
        out.append(1)
    return out


def _plot_max_batch(summary_df: pd.DataFrame, out_png: str, title: str):
    plt.figure()
    for c in sorted(summary_df["chains"].unique()):
        sub = summary_df[summary_df["chains"] == c].sort_values("seq_len")
        plt.plot(sub["seq_len"], sub["max_batch"], marker="o", label=f"chains={c}")
    plt.xlabel("Sequence Length per Chain")
    plt.ylabel("Max Trainable Batch Size")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def run(
    config_json: str = "data/esm2_t33_650M_UR50D.json",
    device: str = "cuda:0",
    seq_lengths=(64, 128, 256, 512),
    chain_counts=(2, 3, 4),
    max_batch_upper: int = 512,
    out_dir: str = "bench_out",
    plot: bool = True,
):
    """
    Find the maximum *trainable* batch size per (seq_len, chains) using ESMWrapper.
    Strategy:
      1) Try powers-of-two upward: 16, 32, 64, ... up to min(seq_len, max_batch_upper).
      2) If none fit, try 8, 4, 2, 1 (downward fallback), filtered by <= seq_len.
    Returns DataFrame with columns: ['seq_len', 'chains', 'max_batch'].
    """
    os.makedirs(out_dir, exist_ok=True)
    dev = _device_ok(device)

    cfg = load_config(config_json)
    args = SimpleNamespace(
        no_multimer=False,
        validate=False,
        print_freq=100,
        wandb=False,
        check_grad=False,
        freeze_self_attn=False,
    )
    esm = ESMWrapper(cfg, args=args).to(dev)
    esm.train()

    rows = []
    for chains in chain_counts:
        for L in seq_lengths:
            cap = min(int(L), int(max_batch_upper))
            if cap < 1:
                rows.append(dict(seq_len=int(L), chains=int(chains), max_batch=0))
                continue

            # 1) Upward pass: 16, 32, 64, ...
            candidates_up = _powers_of_two_up(cap, base=16)

            max_ok = 0
            tried_any = False

            def try_bs(bs: int) -> bool:
                nonlocal esm
                tokens, chain_ids = _make_tokens(bs, L, chains, dev)
                batch = (tokens, chain_ids)
                # keep wrapper's internal log small to avoid accumulation
                if hasattr(esm, "_log"):
                    esm._log = defaultdict(list)
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(dev)
                    torch.cuda.synchronize()
                esm.zero_grad(set_to_none=True)
                ok = False
                try:
                    loss = esm.training_step(batch, batch_idx=0)
                    loss.backward()
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    ok = True
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        ok = False
                    else:
                        raise
                finally:
                    # free tensors promptly
                    del tokens, chain_ids
                return ok

            # Upward sweep
            for bs in candidates_up:
                tried_any = True
                if try_bs(bs):
                    max_ok = bs
                else:
                    break  # larger powers will also OOM

            # 2) If nothing fit in the upward pass, try downward: 8, 4, 2, 1
            if max_ok == 0:
                candidates_down = _powers_of_two_down(cap, start=8)
                for bs in candidates_down:
                    tried_any = True
                    if try_bs(bs):
                        max_ok = bs
                        break  # found the largest in the downward set by construction

            # 3) Edge case: if seq_len < 16 and downward didn’t include seq_len (e.g., seq_len=3)
            # test seq_len itself as a last resort
            if max_ok == 0 and L < 16 and cap >= 1 and cap not in (_powers_of_two_down(cap, start=8)):
                if try_bs(cap):
                    max_ok = cap

            rows.append(dict(seq_len=int(L), chains=int(chains), max_batch=int(max_ok)))
            print(f"[seq_len={L} chains={chains}] max_batch={max_ok} (cap={cap})")

    df = pd.DataFrame(rows).sort_values(["chains", "seq_len"]).reset_index(drop=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"esm_backward_max_batch_{ts}.csv")
    df.to_csv(csv_path, index=False)

    if plot and not df.empty:
        png_path = os.path.join(out_dir, f"esm_backward_max_batch_{ts}.png")
        _plot_max_batch(df, png_path, title="ESMWrapper Backward Pass: Max Trainable Batch Size")
        print(f"[OK] Plot saved: {png_path}")

    print(f"[OK] CSV saved: {csv_path}")
    print(df)
    return df


if __name__ == "__main__":
    fire.Fire(run)



