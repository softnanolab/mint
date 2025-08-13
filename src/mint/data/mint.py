import gzip
import pickle
import random

import torch
from Bio import SeqIO
import lightning as pl
from omegaconf import DictConfig

from mint.data.esm import Alphabet


class CollateFn:
    def __init__(self, truncation_seq_length=None):
        self.alphabet = Alphabet.from_architecture("ESM-1b")
        self.truncation_seq_length = truncation_seq_length
        # self.batch_converter = alphabet.get_batch_converter(truncation_seq_length)

    def __call__(self, batches):
        *chains, contact_masks = zip(*batches)  # unpack: sequences... , then masks
        chains = [self.convert(group) for group in chains]
        chain_ids = [torch.ones(c.shape, dtype=torch.int32) * i for i, c in enumerate(chains)]
        chains = torch.cat(chains, -1)
        chain_ids = torch.cat(chain_ids, -1)
        return chains, chain_ids, contact_masks

    def convert(self, seq_str_list):
        batch_size = len(seq_str_list)
        seq_encoded_list = [
            self.alphabet.encode("<cls>" + seq_str.replace("J", "L") + "<eos>")
            for seq_str in seq_str_list
        ]
        if self.truncation_seq_length:
            for i in range(batch_size):
                seq = seq_encoded_list[i]
                if len(seq) > self.truncation_seq_length:
                    start = random.randint(0, len(seq) - self.truncation_seq_length + 1)
                    seq_encoded_list[i] = seq[start : start + self.truncation_seq_length]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        if self.truncation_seq_length:
            assert max_len <= self.truncation_seq_length
        tokens = torch.empty((batch_size, max_len), dtype=torch.int64)
        tokens.fill_(self.alphabet.padding_idx)

        for i, seq_encoded in enumerate(seq_encoded_list):
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[i, : len(seq_encoded)] = seq
        return tokens


class STRINGDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        links_path,
        seqs_path,
        global_rank=0,
        world_size=1,
        concat=False,
        max_examples=None,
        max_len=None,
        overfit=False,
        seek=None,
    ):
        super().__init__()
        self.links_path = links_path
        self.seqs_path = seqs_path
        self.global_rank = global_rank
        self.world_size = world_size

        if max_examples:
            self.max_iters = int(max_examples // world_size)
        else:
            self.max_iters = None
        self.concat = concat
        self.max_len = max_len
        self.overfit = overfit
        self.seek = seek

    def __len__(self):
        return self.max_iters

    def __iter__(self):
        it = self.__iter_helper__()
        for i, n in enumerate(it):
            if self.seek and i < self.seek:
                pass
            else:
                yield n

    def __iter_helper__(self):
        self.seqs = {}
        links_f = iter(gzip.open(self.links_path, "rt"))
        seqs_f = iter(gzip.open(self.seqs_path, "rt"))
        i, j = 0, 0
        while True:
            try:
                name1, name2 = next(links_f).strip().split()[:2]
                if name1 not in self.seqs:
                    name, seq = next(seqs_f).strip().split()
                    self.seqs[name] = seq
                if name2 not in self.seqs:
                    name, seq = next(seqs_f).strip().split()
                    self.seqs[name] = seq
                if self.max_len and not (
                    len(self.seqs[name1]) <= self.max_len and len(self.seqs[name2]) <= self.max_len
                ):
                    continue  # don't increment i
                if i % self.world_size == self.global_rank:
                    if self.concat:
                        if self.overfit:
                            while True:
                                yield (self.seqs[name1] + "G" * 25 + self.seqs[name2],)
                        else:
                            yield (self.seqs[name1] + "G" * 25 + self.seqs[name2],)
                    else:
                        if self.overfit:
                            while True:
                                yield self.seqs[name1], self.seqs[name2]
                        else:
                            yield self.seqs[name1], self.seqs[name2]
                    j += 1
                i += 1
                if j == self.max_iters:
                    break
            except StopIteration:
                links_f = iter(gzip.open(self.links_path, "rt"))


class ESMDataset(torch.utils.data.IterableDataset):
    def __init__(self, path):
        super().__init__()
        self.seqs = SeqIO.parse(open(path), "fasta")

    def __iter__(self):
        seqs = iter(self.seqs)
        while True:
            seq = next(seqs)
            yield seq.name, seq.seq


"""class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")

    def __len__(self):
        return 1024

    def __getitem__(self, idx):
        return torch.randint(4, 24, size=(1024,))
        seq = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"
        return self.alphabet.get_batch_converter()([("P68871", seq)])[2].squeeze(0)"""


class PseduoMMDataset(torch.utils.data.Dataset):
    """Pseudo Multimer Dataset Class."""

    def __init__(
        self,
        links_path: str,
        seqs_path: str,
        concat: bool = False,
        max_examples: int = None,
        max_len: int = None,
        overfit: bool = False,
    ):
        super().__init__()
        self.links_path = links_path
        self.seqs_path = seqs_path
        self.concat = concat
        self.max_len = max_len
        self.overfit = overfit

        # Load all data upfront for simplicity
        self.data = self._load_data()

        if max_examples:
            self.data = self.data[:max_examples]

    def _load_data(self):
        """Load all data into memory."""
        data = []
        seqs = {}

        # Load sequences first
        with gzip.open(self.seqs_path, "rt") as seqs_f:
            for line in seqs_f:
                name, seq = line.strip().split()
                seqs[name] = seq

        # Load links and create pairs
        with gzip.open(self.links_path, "rt") as links_f:
            for line in links_f:
                name1, name2 = line.strip().split()[:2]

                # Check if both sequences exist and meet length requirements
                if name1 in seqs and name2 in seqs:
                    if self.max_len and (
                        len(seqs[name1]) > self.max_len or len(seqs[name2]) > self.max_len
                    ):
                        continue

                    if self.concat:
                        data.append((seqs[name1] + "G" * 25 + seqs[name2],))
                    else:
                        data.append((seqs[name1], seqs[name2]))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO: think about self.overfit argument here
        if self.overfit:
            # For overfitting, always return the same item
            return self.data[0]
        return self.data[idx]


class PseudoMMDataset(torch.utils.data.Dataset):
    """Pseudo Multimer Dataset Class with Structural Info."""

    def __init__(
        self,
        links_path: str,
        seqs_path: str,
        contact_masks_path: str,
        concat: bool = False,
        max_examples: int = None,
        max_len: int = None,
        overfit: bool = False,
    ):
        super().__init__()
        self.links_path = links_path
        self.seqs_path = seqs_path
        self.contact_masks_path = contact_masks_path
        self.concat = concat
        self.max_len = max_len
        self.overfit = overfit

        # Load all data upfront for simplicity
        self.data = self._load_data()

        if max_examples:
            self.data = self.data[:max_examples]

    def _load_data(self):
        """Load all data into memory."""
        data = []
        seqs = {}

        # Load sequences first
        with gzip.open(self.seqs_path, "rt") as seqs_f:
            for line in seqs_f:
                name, seq = line.strip().split()
                seqs[name] = seq

        # Load contact masks second
        with gzip.open(self.contact_masks_path, "rb") as f:
            contact_masks = pickle.load(f)

        with gzip.open(self.links_path, "rt") as links_f:
            for line in links_f:
                names = line.strip().split()
                if not names:
                    continue

                # Validate presence and optional length constraint
                passed = []
                for n in names:
                    if n not in seqs:
                        passed = []
                        break
                    if self.max_len is not None and len(seqs[n]) > self.max_len:
                        passed = []
                        break
                    passed.append(n)

                # If any failed, skip the whole line
                if len(passed) != len(names):
                    continue

                # All domains must belong to the same chain (chain-level key)
                keys = [n.rsplit("_", 1)[0] for n in passed]
                if len(set(keys)) != 1:
                    continue
                chain_key = keys[0]

                mask = contact_masks.get(chain_key)
                if mask is None:
                    continue

                holder = tuple(seqs[n] for n in passed) + (mask,)
                data.append(holder)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO: think about self.overfit argument here
        if self.overfit:
            # For overfitting, always return the same item
            return self.data[0]
        return self.data[idx]


class PseudoMMDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for STRING protein interaction data."""

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()
        self.config = config

        self.train_dataset = PseudoMMDataset(
            links_path=self.config.train.links_path,
            seqs_path=self.config.train.seqs_path,
            contact_masks_path=self.config.contact_masks_path,
            concat=self.config.train.concat,
            max_examples=self.config.train.max_examples,
            max_len=self.config.train.max_len,
            overfit=self.config.train.overfit,
        )

        # TODO: add val dataset

    def setup(self, stage: str = None):
        """Set up datasets for training and validation."""
        pass

    def train_dataloader(self):
        """Create training data loader."""
        assert self.train_dataset is not None

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=self.config.train.shuffle,
            num_workers=self.config.train.num_workers,
            collate_fn=CollateFn(self.config.train.max_len),
            pin_memory=self.config.train.pin_memory,
        )

    def val_dataloader(self):
        """Create validation data loader."""
        # TODO: add val dataset
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=self.config.train.shuffle,
            num_workers=self.config.train.num_workers,
            collate_fn=CollateFn(self.config.train.max_len),
            pin_memory=self.config.train.pin_memory,
        )


class PseduoMMDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for STRING protein interaction data."""

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()
        self.config = config

        self.train_dataset = PseduoMMDataset(
            links_path=self.config.train.links_path,
            seqs_path=self.config.train.seqs_path,
            concat=self.config.train.concat,
            max_examples=self.config.train.max_examples,
            max_len=self.config.train.max_len,
            overfit=self.config.train.overfit,
        )

        # TODO: add val dataset

    def setup(self, stage: str = None):
        """Set up datasets for training and validation."""
        pass

    def train_dataloader(self):
        """Create training data loader."""
        assert self.train_dataset is not None

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=self.config.train.shuffle,
            num_workers=self.config.train.num_workers,
            collate_fn=CollateFn(self.config.train.max_len),
            pin_memory=self.config.train.pin_memory,
        )

    def val_dataloader(self):
        """Create validation data loader."""
        # TODO: add val dataset
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=self.config.train.shuffle,
            num_workers=self.config.train.num_workers,
            collate_fn=CollateFn(self.config.train.max_len),
            pin_memory=self.config.train.pin_memory,
        )
