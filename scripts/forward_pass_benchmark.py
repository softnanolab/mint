
import time
import json
import math
import os
from datetime import datetime

import fire
import torch
import pandas as pd
import matplotlib.pyplot as plt

from mint.helpers.extract import load_config, MINTWrapper


def _device_ok(device: str):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use device='cpu' or enable GPU.")
    return torch.device(device)


def _make_dummy_tokens(batch_size: int, seq_len: int, chains: int, device: torch.device):
    """
    Create concatenated token and chain_id tensors that MINTWrapper accepts:
      tokens: (B, sum(chain_lengths)) longs in [1, 32)
      chain_ids: (B, sum(chain_lengths)) longs with per-token chain index [0..chains-1]
    """
    toks = []
    ids = []
    for c in range(chains):
        toks.append(torch.randint(1, 32, (batch_size, seq_len), dtype=torch.long))
        ids.append(torch.full((batch_size, seq_len), c, dtype=torch.long))
    tokens = torch.cat(toks, dim=1).to(device)
    chain_ids = torch.cat(ids, dim=1).to(device)
    return tokens, chain_ids


def _plot_lines(df: pd.DataFrame, x: str, y: str, group: str, title: str, xlabel: str, ylabel: str, out_png: str):
    plt.figure()
    for g in sorted(df[group].unique()):
        sub = df[df[group] == g].sort_values(x)
        plt.plot(sub[x], sub[y], marker="o", label=f"{group}={g}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def run(
    config_json: str = "data/esm2_t33_650M_UR50D.json",
    checkpoint_path: str = "mint.ckpt",
    device: str = "cuda:0",
    seq_lengths=(64, 128, 256, 512),
    chain_counts=(2, 3, 4),
    batch_size: int = 1,
    sep_chains: bool = True,
    out_dir: str = "bench_out",
):
    """
    Forward-pass (inference) benchmark for MINTWrapper.
    Saves: CSV with results + two PNG plots (time vs len, memory vs len), grouped by chain count.
    """
    os.makedirs(out_dir, exist_ok=True)
    dev = _device_ok(device)

    cfg = load_config(config_json)
    wrapper = MINTWrapper(cfg, checkpoint_path, sep_chains=sep_chains, device=dev)
    wrapper.eval()

    results = []
    for chains in chain_counts:
        for L in seq_lengths:
            tokens, chain_ids = _make_dummy_tokens(batch_size, L, chains, dev)

            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            with torch.no_grad():
                _ = wrapper(tokens, chain_ids)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            peak_mb = 0.0
            if dev.type == "cuda":
                peak_mb = torch.cuda.max_memory_allocated(dev) / 1e6

            results.append(
                dict(
                    chains=int(chains),
                    seq_len=int(L),
                    batch_size=int(batch_size),
                    time_s=float(elapsed),
                    peak_mem_mb=float(peak_mb),
                )
            )

    df = pd.DataFrame(results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"forward_bench_{ts}.csv")
    df.to_csv(csv_path, index=False)

    _plot_lines(
        df,
        x="seq_len",
        y="peak_mem_mb",
        group="chains",
        title="Forward Pass: Peak GPU Memory vs Sequence Length",
        xlabel="Sequence Length per Chain",
        ylabel="Peak Memory (MB)",
        out_png=os.path.join(out_dir, f"forward_memory_{ts}.png"),
    )
    _plot_lines(
        df,
        x="seq_len",
        y="time_s",
        group="chains",
        title="Forward Pass: Time vs Sequence Length",
        xlabel="Sequence Length per Chain",
        ylabel="Time (s)",
        out_png=os.path.join(out_dir, f"forward_time_{ts}.png"),
    )

    print(f"[OK] Saved: {csv_path}")
    print(f"[OK] Plots: forward_memory_{ts}.png, forward_time_{ts}.png in {out_dir}")


if __name__ == "__main__":
    fire.Fire(run)
