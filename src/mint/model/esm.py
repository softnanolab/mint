# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union
import torch
import torch.nn as nn

from mint.data.esm import Alphabet
from mint.model.modules import ESM1bLayerNorm, RobertaLMHead, TransformerLayer


class ESM2(nn.Module):

    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union["Alphabet", str] = "ESM-1b",
        token_dropout: bool = True,
        use_multimer: bool = False,
        use_mlp_head: bool = False,
        mlp_hidden: int = 128,
        mlp_out_dim: int | None = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, Alphabet):
            alphabet = Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout
        self.use_multimer = use_multimer
        self.use_mlp_head = use_mlp_head
        self.mlp_out_dim = mlp_out_dim
        self._init_submodules()
        if self.use_mlp_head:
            if self.mlp_out_dim is None:
                raise ValueError("mlp_out_dim must be provided when use_mlp_head=True")
            # Build MLP head
            self.mlp_head = nn.Sequential(
                nn.Linear(self.embed_dim, mlp_hidden),
                nn.Sigmoid(),
                nn.Linear(mlp_hidden, self.mlp_out_dim),
                nn.Sigmoid(),
            )
            # Disable MLM head
            self.lm_head = None
        else:
            self.mlp_head = None

    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.embed_dim, padding_idx=self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                    use_multimer=self.use_multimer,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)
        # Only create lm_head if not using MLP head
        if not getattr(self, "use_mlp_head", False):
            self.lm_head = RobertaLMHead(
                embed_dim=self.embed_dim,
                output_dim=self.alphabet_size,
                weight=self.embed_tokens.weight,
            )

    def disable_mlp_head(self):
        self.mlp_head = None
        self.use_mlp_head = False
        if self.lm_head is None:
            # Recreate MLM head if needed
            self.lm_head = RobertaLMHead(
                embed_dim=self.embed_dim,
                output_dim=self.alphabet_size,
                weight=self.embed_tokens.weight,
            )

    def forward(
        self,
        tokens,
        chain_ids=None,
        repr_layers=[],
        need_head_weights=False,
        return_contacts=False,
        return_mlp: bool = False,
    ):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        if chain_ids is None:
            chain_ids = torch.zeros_like(tokens)
        self_attn_mask = ~torch.eq(chain_ids.unsqueeze(-1), chain_ids.unsqueeze(-2))  # B, T, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
                self_attn_mask=self_attn_mask,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x

        mlp_out = None
        if self.use_mlp_head and self.mlp_head is not None:
            mlp_out = self.mlp_head(x)  # (B, T, mlp_out_dim)

        logits = None
        if (not self.use_mlp_head) and self.lm_head is not None:
            logits = self.lm_head(x)

        result = {
            "representations": hidden_representations,
        }
        if logits is not None:
            result["logits"] = logits
        if mlp_out is not None and return_mlp:
            result["mlp_out"] = mlp_out
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]
