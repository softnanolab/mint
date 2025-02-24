import logging, os
import re
from abc import ABC, abstractmethod
from functools import partial
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
import tqdm as tqdm
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BatchEncoding,
    DefaultDataCollator,
    T5EncoderModel,
    T5Tokenizer
)
from datasets import Dataset
from transformers.modeling_outputs import BaseModelOutput

def pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor, pool_type: str
) -> torch.Tensor:
    """Pool embeddings across the sequence length dimension."""
    assert (
        last_hidden_states.ndim == 3
    ), f"Expected hidden_states to have shape [batch, seq_len, D], got shape: {last_hidden_states.shape}"
    assert (
        attention_mask.ndim == 2
    ), f"Expected attention_mask to have shape [batch, seq_len], got shape: {attention_mask.shape}"
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    if pool_type == "mean":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "max":
        emb = last_hidden.max(dim=1)[0]
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        emb = last_hidden[torch.arange(last_hidden.size(0)), attention_mask.sum(1) - 1]
    else:
        raise ValueError(f"pool_type {pool_type} not supported")
    return emb

logger = logging.getLogger(__name__)

class BioSeqTransformer(ABC):

    def __init__(
        self,
        model_name: str,
        layers,
        devices: List[int] = [0],
        num_processes: int = 16,
        max_seq_length: int = 2048,
        l2_norm: bool = False,
        batch_size: int = 128,
        pool_type: str = "mean",
    ):
        super().__init__()

        os.environ['HF_HOME'] = './'

        self.id = self.__class__.__name__
        self.hf_name = model_name
        self.encoder = self._load_model(model_name)
        if not hasattr(self.encoder, "config"):
            raise ValueError(
                'The model from `self._load_model()` must have a "config" attribute.'
            )
        self.config = self.encoder.config
        self.tokenizer = self._get_tokenizer(model_name)
        self.num_param = sum(p.numel() for p in self.encoder.parameters())
        self.data_collator = DefaultDataCollator()
        self.gpu_count = len(devices)
        self.l2_norm = l2_norm

        self.device = torch.device(
            f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu"
        )
        self.num_processes = num_processes
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.pool_type = pool_type

        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder, device_ids=devices)
        self.encoder.to(self.device)
        self.encoder.eval()

        mid_layer = self.num_layers // 2
        last_layer = self.num_layers - 1
        mid_layer_label = f"mid ({mid_layer})"
        last_layer_label = f"last ({self.num_layers - 1})"

        if layers is None:
            logger.debug(f"Using default layers: {mid_layer_label}, {last_layer_label}")
            self.layers = [mid_layer, last_layer]
            self.layer_labels = [mid_layer_label, last_layer_label]
        elif layers == "mid":
            self.layers = [mid_layer]
            self.layer_labels = [mid_layer_label]
        elif layers == "last":
            self.layers = [last_layer]
            self.layer_labels = [last_layer_label]
        else:
            self.layers = layers
            self.layer_labels = [str(layer) for layer in layers]

    def _encode_single_batch(self, batch_dict: Dict[str, Tensor]):
        """Returns the output embedding for the given batch with shape [batch, num_layers, D]."""
        outputs = self.encoder(**batch_dict, output_hidden_states=True)
        embeds = [outputs.hidden_states[layer] for layer in self.layers]
        embeds = [
            pool(layer_embeds, batch_dict["attention_mask"], self.pool_type)
            for layer_embeds in embeds
        ]
        # Stack with shape [B, num_layers, D].
        embeds = torch.stack(embeds, dim=1)
        return embeds

    def _load_model(self, model_name):
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, cache_dir='./baseline_models/')
        print(f'Loaded model for: {model_name}')
        return model

    def _get_tokenizer(self, model_name):
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir='./baseline_models/')
        print(f'Loaded tokenizer for: {model_name}')
        return tok

    def _tokenize_func(
        self, tokenizer, examples: Dict[str, List], max_seq_length: int
    ) -> BatchEncoding:
        batch_dict = tokenizer(
            examples["input_seqs"],
            max_length=max_seq_length,
            padding=True,
            truncation=True,
        )
        return batch_dict

    @property
    def metadata(self) -> Dict:
        return {
            "hf_name": self.hf_name,
            "num_layers": self.num_layers,
            "num_params": self.num_param,
            "embed_dim": self.embed_dim,
        }

    @property
    @abstractmethod
    def num_layers(self) -> int:
        pass

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        pass

    @torch.no_grad()
    def encode(self, sequences, **kwargs):
        """Returns a list of embeddings for the given sequences.
        Args:
            sequences (`List[str]`): List of sequences to encode
        Returns:
            `np.ndarray`: Embeddings for the given sequences of shape [num_sequences, num_layers, embedding_dim].
        """
        dataset = Dataset.from_dict({"input_seqs": sequences})
        dataset.set_transform(
            partial(
                self._tokenize_func, self.tokenizer, max_seq_length=self.max_seq_length
            )
        )
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size * self.gpu_count,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_processes,
            collate_fn=self.data_collator,
            pin_memory=True,
        )

        if max(self.layers) >= self.num_layers:
            raise ValueError(
                f"Layer {max(self.layers)} is not available in the model. Choose a layer between 0 and {self.num_layers - 1}"
            )

        encoded_embeds = []
        for batch_dict in tqdm.tqdm(
            data_loader, desc="encoding", mininterval=10
        ):
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            embeds = self._encode_single_batch(batch_dict)

            if self.l2_norm:
                embeds = F.normalize(embeds, p=2, dim=-1)
            encoded_embeds.append(embeds.cpu())

        return torch.cat(encoded_embeds, dim=0)

    @torch.no_grad()
    def encode_two(self, sequences1, sequences2, how='subtract', **kwargs):
        encodings1 = self.encode(sequences1)
        encodings2 = self.encode(sequences2)

        if how == 'subtract':
            return encodings1 - encodings2
        else:
            return torch.cat([encodings1, encodings2], -1)
        

class ESM(BioSeqTransformer):
    """ESM model from https://huggingface.co/docs/transformers/en/model_doc/esm"""

    MODEL_NAMES = [
        "facebook/esm2_t6_8M_UR50D",
        "facebook/esm2_t12_35M_UR50D",
        "facebook/esm2_t30_150M_UR50D",
        "facebook/esm2_t33_650M_UR50D",
        "facebook/esm2_t36_3B_UR50D",
        "facebook/esm2_t48_15B_UR50D",
    ]

    @property
    def num_layers(self) -> int:
        return self.config.num_hidden_layers

    @property
    def embed_dim(self) -> int:
        return self.config.hidden_size

class ProtT5(BioSeqTransformer):
    """ProtT5 model from https://github.com/agemagician/ProtTrans"""

    MODEL_NAMES = [
        "Rostlab/prot_t5_xl_uniref50",
        "Rostlab/prot_t5_xl_bfd",
        "Rostlab/prot_t5_xxl_uniref50",
        "Rostlab/prot_t5_xxl_bfd",
    ]


    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    @property
    def embed_dim(self) -> int:
        return self.config.d_model

    def _load_model(self, model_name):
        model = T5EncoderModel.from_pretrained(model_name, cache_dir='./baseline_models/')
        print(f'Loaded model for: {model_name}')
        return model

    def _get_tokenizer(self, model_name):
        tok = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, cache_dir='./baseline_models/')
        print(f'Loaded tokenizer for: {model_name}')
        return tok

    def _tokenize_func(
        self, tokenizer, examples: Dict[str, List], max_seq_length: int
    ) -> BatchEncoding:
        example_sequences = examples["input_seqs"]
        # Add space between amino acids to make sure they are tokenized correctly.
        example_sequences = [" ".join(sequence) for sequence in example_sequences]
        example_sequences = [
            re.sub(r"[UZOB]", "X", sequence) for sequence in example_sequences
        ]
        batch_dict = tokenizer(
            example_sequences,
            max_length=max_seq_length,
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )

        return batch_dict

class ProGen(BioSeqTransformer):
    """ProGen models from https://github.com/salesforce/progen."""

    MODEL_NAMES = [
        "hugohrban/progen2-small",
        "hugohrban/progen2-medium",
        "hugohrban/progen2-base",
        "hugohrban/progen2-large",
        "hugohrban/progen2-xlarge",
    ]


    @property
    def num_layers(self) -> int:
        return self.config.n_layer

    @property
    def embed_dim(self) -> int:
        return self.config.embed_dim

    def _load_model(self, model_name):
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, cache_dir='./baseline_models/')
        print(f'Loaded model for: {model_name}')
        return model

    def _get_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir='./baseline_models/'
        )
        tokenizer.pad_token = "<|pad|>"
        print(f'Loaded tokenizer for: {model_name}')
        return tokenizer

    def _encode_single_batch(self, batch_dict: Dict[str, Tensor]):
        """Returns the output embedding for the given batch with shape [batch, num_layers, D]."""
        outputs: BaseModelOutput = self.encoder(
            input_ids=batch_dict["input_ids"],
            output_hidden_states=True,
            use_cache=False,
        )
        embeds = [outputs.hidden_states[layer] for layer in self.layers]
        embeds = [
            pool(layer_embeds, batch_dict["attention_mask"], self.pool_type)
            for layer_embeds in embeds
        ]
        # Stack with shape [B, num_layers, D].
        embeds = torch.stack(embeds, dim=1)
        return embeds

# model = ProtT5(model_name='Rostlab/prot_t5_xl_bfd', layers='last', devices=[0], batch_size=2)
# model = ProtT5(model_name='Rostlab/prot_t5_xl_uniref50', layers='last', devices=[0], batch_size=2)
# model = ProGen(model_name='hugohrban/progen2-xlarge', layers='last', devices=[0], batch_size=2)
# model = ProGen(model_name='hugohrban/progen2-large', layers='last', devices=[0], batch_size=2)
# model = ESM(model_name='facebook/esm2_t36_3B_UR50D', layers='last', devices=[0], batch_size=2)
# model = ESM(model_name='facebook/esm2_t33_650M_UR50D', layers='last', devices=[0], batch_size=2)
# model = ESM(model_name='facebook/esm2_t30_150M_UR50D', layers='last', devices=[0], batch_size=2)
# model = ESM(model_name='facebook/esm1b_t33_650M_UR50S', layers='last', devices=[0], batch_size=2)
