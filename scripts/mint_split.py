#!/usr/bin/env python3
import os
import re
import shutil
import subprocess as sp
import sys
import tempfile
from pathlib import Path
from random import Random

# ---------- Utilities ----------

def install_mmseqs():
    print("[i] MMseqs2 not found — attempting to install...")
    try:
        sp.run(["conda", "install", "-y", "-c", "conda-forge", "-c", "bioconda", "mmseqs2"], check=True)
        print("[i] Installed MMseqs2 via conda.")
        return "mmseqs"
    except Exception:
        print("[!] Conda install failed. Falling back to direct binary download.")
        tmpdir = Path(tempfile.mkdtemp(prefix="mmseqs_bin_"))
        tar_path = tmpdir / "mmseqs.tar.gz"
        url = "https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz"
        sp.run(["wget", "-q", "-O", str(tar_path), url], check=True)
        sp.run(["tar", "xzf", str(tar_path), "-C", str(tmpdir)], check=True)
        mmseqs_bin = tmpdir / "mmseqs/bin/mmseqs"
        if not mmseqs_bin.exists():
            sys.exit("Error: could not find mmseqs binary after extraction.")
        os.chmod(mmseqs_bin, 0o755)
        print(f"[i] Downloaded mmseqs binary to: {mmseqs_bin}")
        return str(mmseqs_bin)

def find_mmseqs():
    try:
        sp.run(["mmseqs", "--version"], check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        return "mmseqs"
    except Exception:
        return install_mmseqs()

def detect_real_cpu_count():
    try:
        return int(sp.check_output(["nproc"]).strip())
    except Exception:
        return os.cpu_count() or 1

def run(cmd):
    sp.run(cmd, check=True)

def normalize_to_oneline_fasta(src: Path, dst: Path):
    first_line = ""
    with src.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                first_line = line
                break
    is_fa = first_line.startswith(">")

    with src.open("r", encoding="utf-8", errors="ignore") as fin, dst.open("w", encoding="utf-8") as fout:
        if is_fa:
            seq, header = [], None
            for raw in fin:
                s = raw.rstrip("\n\r")
                if s.startswith(">"):
                    if header is not None:
                        fout.write(f">{header}\n{''.join(seq)}\n")
                    header = s[1:].strip().split()[0]
                    seq = []
                else:
                    seq.append(re.sub(r"[ \t]", "", s))
            if header is not None:
                fout.write(f">{header}\n{''.join(seq)}\n")
        else:
            for raw in fin:
                s = raw.strip()
                if not s or s.startswith("#"):
                    continue
                parts = re.split(r"[ \t]+", s, maxsplit=1)
                if len(parts) != 2:
                    sys.exit(f"Bad line (need 'ID SEQ'): {s}")
                pid, seq = parts[0], re.sub(r"[ \t]", "", parts[1])
                fout.write(f">{pid}\n{seq}\n")

def fasta_to_dict(fasta_path: Path) -> dict:
    out = {}
    with fasta_path.open() as f:
        cur = None
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith(">"):
                cur = line[1:].split()[0]
                out[cur] = ""
            else:
                out[cur] += re.sub(r"[ \t]", "", line)
    return out

# ---------- Main ----------

def main():
    seqs_path = Path(input("Path to sequences file (e.g., training.seqs.txt): ").strip())
    links_path = Path(input("Path to links file (e.g., training.links.txt): ").strip())
    split_ratio = float(input(
        "What fraction of CLUSTERS (≥40% similar sequences) should go into the training set? (default 0.9): "
    ) or "0.9")

    threads = detect_real_cpu_count()
    out_train_seqs = Path("train.seqs.txt")
    out_val_seqs   = Path("val.seqs.txt")
    out_train_links = Path("train.links.txt")
    out_val_links   = Path("val.links.txt")
    keep_temp = False
    seed = None

    if not seqs_path.exists():
        sys.exit(f"Missing sequences file: {seqs_path}")
    if not links_path.exists():
        sys.exit(f"Missing links file: {links_path}")

    mmseqs_bin = find_mmseqs()
    workdir = Path(tempfile.mkdtemp(prefix="mint_split_tmp_"))
    tmpdir = workdir / "mmseqs_tmp"
    tmpdir.mkdir(exist_ok=True)

    try:
        print("[0/8] Normalizing input → one-line FASTA")
        fa = workdir / "training.seqs.fasta"
        normalize_to_oneline_fasta(seqs_path, fa)

        print(f"[1/8] mmseqs createdb (threads auto={threads})")
        run([mmseqs_bin, "createdb", str(fa), str(workdir / "seqDB")])

        print("[2/8] mmseqs linclust (≥40% identity)")
        run([
            mmseqs_bin, "linclust", str(workdir / "seqDB"), str(workdir / "clu40"),
            str(tmpdir), "--min-seq-id", "0.4", "--cov-mode", "1", "-c", "0.8",
            "--threads", str(threads)
        ])

        print("[3/8] mmseqs createtsv")
        clu_tsv = workdir / "clu40.tsv"
        run([mmseqs_bin, "createtsv", str(workdir / "seqDB"), str(workdir / "seqDB"),
             str(workdir / "clu40"), str(clu_tsv)])

        cluster_sizes = {}
        with clu_tsv.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rep, member = line.strip().split("\t")[:2]
                cluster_sizes.setdefault(rep, 0)
                cluster_sizes[rep] += 1

        num_clusters = len(cluster_sizes)

        print("[4/8] Splitting by clusters")
        reps = sorted(cluster_sizes.keys())
        rng = Random(seed)
        rng.shuffle(reps)
        n_train = max(1, min(len(reps) - 1, int(len(reps) * split_ratio)))
        reps_train, reps_val = set(reps[:n_train]), set(reps[n_train:])

        train_ids, val_ids = set(), set()
        with clu_tsv.open() as f:
            for line in f:
                r, m = line.strip().split("\t")[:2]
                if r in reps_train:
                    train_ids.add(m)
                elif r in reps_val:
                    val_ids.add(m)

        print("[5/8] Verifying disjoint ID sets")
        if train_ids & val_ids:
            overlap = list(train_ids & val_ids)[:3]
            sys.exit(f"Train/Val ID overlap detected (e.g., {overlap}).")

        print("[6/8] Writing train/val sequences (ID<TAB>SEQ)")
        seq_dict = fasta_to_dict(fa)
        with out_train_seqs.open("w") as ftr:
            for pid in sorted(train_ids):
                if pid in seq_dict:
                    ftr.write(f"{pid}\t{seq_dict[pid]}\n")
        with out_val_seqs.open("w") as fvr:
            for pid in sorted(val_ids):
                if pid in seq_dict:
                    fvr.write(f"{pid}\t{seq_dict[pid]}\n")

        print("[7/8] Splitting links (intra-split only)")
        def write_links(keep_ids: set, out_path: Path):
            with links_path.open() as fin, out_path.open("w") as fout:
                for line in fin:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    a, b, *_ = re.split(r"[ \t]+", s)
                    if a in keep_ids and b in keep_ids:
                        fout.write(f"{a}\t{b}\n")
        write_links(train_ids, out_train_links)
        write_links(val_ids, out_val_links)

        train_seq_count = sum(1 for _ in out_train_seqs.open())
        val_seq_count   = sum(1 for _ in out_val_seqs.open())
        train_link_count = sum(1 for _ in out_train_links.open())
        val_link_count   = sum(1 for _ in out_val_links.open())

        print("[8/8] Done.")
        print(f"  Train: {train_seq_count} seqs, {train_link_count} links")
        print(f"  Val  : {val_seq_count} seqs, {val_link_count} links")
        print(f"  Total clusters formed (≥40% identity): {num_clusters}")

    finally:
        if not keep_temp:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"[i] Temp preserved at: {workdir}")

if __name__ == "__main__":
    main()
