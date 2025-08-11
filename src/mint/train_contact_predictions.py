"""
Supervised training (row-wise contact mask) for ESM2 with a row MLP head.

Task:
For each residue i produce a length-(sequence_length) vector y_i where
  target[i,j] =
      -1  (SAME chain)
       0  (DIFFERENT chain & CA distance > 5.0 Å)
       1  (DIFFERENT chain & CA distance <= 5.0 Å)

Implementation:
- We train a 3‑class classifier (same_chain / far / close).
- Mapping during training:
      -1 -> class 0   (same_chain)
       0 -> class 1   (far)
       1 -> class 2   (close)
- At inference you can map predicted argmax back to {-1,0,1}.
- The MLP head produces (B, T, T, 3) but is factored “row-wise”:
    For each residue embedding h_i we output logits for all j positions.

Data format (prepare before training):
data_root/
  index.txt                # one sample id per line
  sequences/{id}.fasta     # sequence (with or without header)
  labels/{id}.npz          # contains:
       chain_ids  : int32 array shape (L,)      (e.g. 0,0,0,1,1,...)
       ca_coords  : float32 array shape (L,3)   (Cα coordinates in Å)
Notes:
- We derive the distance matrix on the fly from ca_coords.
- If a residue lacks coordinates you can store NaNs; they will be masked.

Run:
python -m mint.train_row_contact --data_dir DATA --max_len 512 --epochs 5

"""

from __future__ import annotations
import argparse
from pathlib import Path
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as pl

from mint.model.esm import ESM2  # Your modified ESM2 that supports turning off lm_head
from scripts.contact_predictions import build_contact_mask, locate_cif_file, load_structure  # reuse


# ---------------- Head (row-wise MLP) ---------------- #

class RowContactMLP(nn.Module):
    """
    Row-wise MLP: for each residue embedding h_i -> logits over all positions j.

    Produces logits shape (B, T, T, classes). Requires a fixed max_len.
    """
    def __init__(self, embed_dim: int, hidden: int, max_len: int, classes: int = 3):
        super().__init__()
        self.max_len = max_len
        self.classes = classes
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max_len * classes),
        )

    def forward(self, h: torch.Tensor, seq_mask: torch.Tensor):
        """
        h: (B,T,E)
        seq_mask: (B,T) bool (True = valid position)
        Returns logits: (B,T,T,classes)
        """
        B, T, E = h.shape
        logits_flat = self.net(h)                # (B,T, max_len*classes)
        logits = logits_flat.view(B, T, self.classes, self.max_len).transpose(2, 3)  # (B,T,max_len,classes)
        # Mask columns beyond actual sequence length
        # seq_mask: (B,T) -> col_mask (B,1,T,1)
        col_mask = seq_mask.unsqueeze(1).unsqueeze(-1)  # (B,1,T,1)
        logits = logits[:, :, :T, :]                   # trim to current T
        logits = logits.masked_fill(~col_mask, 0.0)
        return logits  # (B,T,T,classes)


# ---------------- Dataset ---------------- #

class RowContactDataset(Dataset):
    """
    Loads sequence + chain_ids + CA coords.
    We build targets inside collate.
    """
    def __init__(self, root: Path, max_len: int):
        self.root = Path(root)
        self.seq_dir = self.root / "sequences"
        self.lab_dir = self.root / "labels"
        self.ids = [l.strip() for l in (self.root / "index.txt").read_text().splitlines() if l.strip()]
        self.max_len = max_len

    def _load_fasta(self, fp: Path) -> str:
        lines = fp.read_text().splitlines()
        if lines and lines[0].startswith(">"):
            lines = lines[1:]
        return "".join(lines).strip()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        seq = self._load_fasta(self.seq_dir / f"{sid}.fasta")
        lab = np.load(self.lab_dir / f"{sid}.npz")
        chain_ids = lab["chain_ids"].astype(np.int32)        # (L,)
        ca_coords = lab["ca_coords"].astype(np.float32)      # (L,3)
        return sid, seq, chain_ids, ca_coords


# ---------------- Collate ---------------- #

def build_targets(chain_ids: torch.Tensor, dist_mat: torch.Tensor, cutoff: float):
    """
    chain_ids: (L,)
    dist_mat : (L,L) (float) distances (NaN or large for invalid)
    cutoff   : distance threshold
    Returns target (L,L) with values in {-1,0,1}
    """
    same = chain_ids.unsqueeze(0) == chain_ids.unsqueeze(1)           # (L,L)
    # Mark invalid distances (NaN) as far for now
    invalid = torch.isnan(dist_mat)
    close = (~same) & (~invalid) & (dist_mat <= cutoff)
    far = (~same) & (~invalid) & (dist_mat > cutoff)
    target = torch.zeros_like(dist_mat).to(torch.int8)
    target[same] = -1
    target[far] = 0
    target[close] = 1
    return target  # (L,L)


def pairwise_ca_dist(ca: torch.Tensor):
    # ca: (L,3) with possible NaNs
    diff = ca.unsqueeze(1) - ca.unsqueeze(0)  # (L,L,3)
    dist = torch.linalg.vector_norm(diff, dim=-1)  # (L,L)
    dist[torch.isnan(dist)] = math.inf
    return dist


def collate(batch, alphabet, max_len: int, device="cpu"):
    """
    Produces:
      tokens: (B, T+2)  (CLS + seq + EOS)
      seq_mask: (B, T)  (True for valid residues, excluding CLS/EOS)
      target_rows: (B, T, T)  with {-1,0,1} padded with -2 beyond length
    """
    ids, seqs, chain_lists, ca_lists = zip(*batch)
    to_idx = alphabet.get_idx
    pad_idx = alphabet.padding_idx
    cls_idx = alphabet.cls_idx
    eos_idx = alphabet.eos_idx

    B = len(seqs)
    Tm = max_len

    tokens = torch.full((B, Tm + 2), pad_idx, dtype=torch.long)
    seq_mask = torch.zeros((B, Tm), dtype=torch.bool)
    targets = torch.full((B, Tm, Tm), -2, dtype=torch.int8)  # -2 = padding ignore

    for b, (seq, chains, ca) in enumerate(zip(seqs, chain_lists, ca_lists)):
        seq = seq[:Tm]
        L = len(seq)
        chains = torch.from_numpy(chains)[:L]
        ca = torch.from_numpy(ca)[:L]
        tok = [cls_idx] + [to_idx(a) for a in seq] + [eos_idx]
        tokens[b, : L + 2] = torch.tensor(tok)
        seq_mask[b, :L] = True
        # distances
        dist = pairwise_ca_dist(ca)
        tgt = build_targets(chains, dist, cutoff)  # (L,L)
        targets[b, :L, :L] = tgt

    return {
        "ids": ids,
        "tokens": tokens.to(device),
        "seq_mask": seq_mask.to(device),
        "targets": targets.to(device),  # {-1,0,1,-2}
    }


def build_targets_from_domains(domain_coord_lists, device, cutoff):
    """
    domain_coord_lists: list of (Li,3) arrays (per domain, ordered along sequence)
    Returns (L,L) int8 mask with {-1,0,1} matching build_contact_mask.
    """
    mask = build_contact_mask(domain_coord_lists, cutoff=cutoff)  # numpy (N,N)
    return torch.from_numpy(mask).to(device)


# In collate(), if you have domain boundaries + coords per domain, replace the distance logic:
        # distances / chain-based targets (old):
        # dist = pairwise_ca_dist(ca)
        # tgt = build_targets(chains, dist, cutoff)

        # New (domain-based) example:
        # domain_coord_lists = [...]  # build this from your metadata aligning to sequence order
        # tgt = build_targets_from_domains(domain_coord_lists, tokens.device, cutoff)


# ---------------- Lightning Module ---------------- #

class RowContactModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        # Base model (disable LM head; we will add our row head)
        self.esm = ESM2(
            num_layers=args.layers,
            embed_dim=args.embed_dim,
            attention_heads=args.heads,
            token_dropout=False,
            use_multimer=False,
            use_mlp_head=False,  # keep base unchanged; we add custom row head below
        )
        self.head = RowContactMLP(
            embed_dim=args.embed_dim,
            hidden=args.row_hidden,
            max_len=args.max_len,
            classes=3,
        )
        self.pad_idx = self.esm.padding_idx

    def forward(self, tokens):
        # Get embeddings (final hidden states)
        out = self.esm(tokens, repr_layers=[], need_head_weights=False, return_mlp=False)
        # The ESM2 forward (your implementation) should output final hidden states in out["representations"]
        # If not, adapt: capture 'x' before lm_head and return it.
        # For simplicity assume final embedding is last layer norm output returned as out["representations"][last_layer]
        # If absent, modify ESM2 to include final "embedding" key.
        if out["representations"]:
            # take highest layer index
            layer_idx = max(out["representations"].keys())
            h = out["representations"][layer_idx]  # (B,T,E)
        else:
            raise RuntimeError("ESM2 did not return representations; ensure repr_layers is set or modify ESM2.")
        return h  # (B,T,E)

    def training_step(self, batch, _):
        tokens = batch["tokens"]
        seq_mask = batch["seq_mask"]
        targets = batch["targets"]  # (B,T,T) {-1,0,1,-2}

        h = self.forward(tokens)[:, 1:-1, :]  # strip CLS/EOS -> (B,T,E) where T = max_len actually used
        seq_mask_inner = seq_mask  # already excludes CLS/EOS
        logits = self.head(h, seq_mask_inner)  # (B,T,T,3)

        # Prepare labels
        # Map {-1,0,1} -> {0,1,2}; padding -2 -> ignore
        labels = targets.clone()
        labels[labels == -1] = 0
        labels[labels == 0] = 1
        labels[labels == 1] = 2
        ignore_index = -100
        labels[labels == -2] = ignore_index

        loss = F.cross_entropy(
            logits.view(-1, 3),
            labels.view(-1),
            ignore_index=ignore_index,
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        tokens = batch["tokens"]
        seq_mask = batch["seq_mask"]
        targets = batch["targets"]
        h = self.forward(tokens)[:, 1:-1, :]
        logits = self.head(h, seq_mask)
        labels = targets.clone()
        labels[labels == -1] = 0
        labels[labels == 0] = 1
        labels[labels == 1] = 2
        labels[labels == -2] = -100
        loss = F.cross_entropy(
            logits.view(-1, 3),
            labels.view(-1),
            ignore_index=-100,
        )
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr)


# ---------------- CLI ---------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--embed_dim", type=int, default=768)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--row_hidden", type=int, default=512)
    p.add_argument("--dist_cutoff", type=float, default=5.0)
    return p.parse_args()


def main():
    args = parse_args()
    tmp = ESM2(num_layers=1, embed_dim=64, attention_heads=4, use_mlp_head=False)
    alphabet = tmp.alphabet

    ds = RowContactDataset(args.data_dir, args.max_len)

    def _collate(b):
        return collate(b, alphabet, args.max_len, args.dist_cutoff, device="cpu")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )
    val_loader = loader  # (split for real use)

    model = RowContactModule(args)
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto", devices="auto", log_every_n_steps=10)
    trainer.fit(model, loader, val_loader)


if __name__ == "__main__":
    main()