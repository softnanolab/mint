# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mint.model.multihead_attention import MultiheadAttention  # noqa


def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


class ESM1LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, affine=True):
        """Construct a layernorm layer in the TF style (eps inside the sqrt)."""
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight, self.bias = None, None

    def forward(self, x):
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keepdim=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keepdim=True)
        x = x_zeromean / torch.sqrt(variances + self.eps)
        if self.affine:
            x = (self.weight * x) + self.bias
        return x


try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    class ESM1bLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)


except ImportError:
    from torch.nn import LayerNorm as ESM1bLayerNorm


class TransformerLayer(nn.Module):
    """Transformer layer block."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        add_bias_kv=True,
        use_esm1b_layer_norm=False,
        use_rotary_embeddings: bool = False,
        use_multimer=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.use_rotary_embeddings = use_rotary_embeddings
        self.use_multimer = use_multimer
        self._init_submodules(add_bias_kv, use_esm1b_layer_norm)

    def _init_submodules(self, add_bias_kv, use_esm1b_layer_norm):
        BertLayerNorm = ESM1bLayerNorm if use_esm1b_layer_norm else ESM1LayerNorm

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )

        if self.use_multimer:
            self.multimer_attn = MultiheadAttention(
                self.embed_dim,
                self.attention_heads,
                add_bias_kv=add_bias_kv,
                add_zero_attn=False,
                use_rotary_embeddings=False,
                no_proj=True,
            )

        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)

    def forward(
        self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        if self.use_multimer:
            self_attn, self_v = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                before_softmax=True,
            )
            multimer_attn, multimer_v = self.multimer_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                before_softmax=True,
            )
            attn_weights = torch.where(self_attn_mask.unsqueeze(1), multimer_attn, self_attn)
            attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
            attn_probs_dropout = F.dropout(
                attn_probs, p=self.self_attn.dropout, training=self.training
            )
            self_attn_probs = attn_probs_dropout.masked_fill(self_attn_mask.unsqueeze(1), 0.0)
            multimer_attn_probs = attn_probs_dropout.masked_fill(~self_attn_mask.unsqueeze(1), 0.0)
            attn_out = torch.matmul(self_attn_probs, self_v) + torch.matmul(
                multimer_attn_probs, multimer_v
            )

            attn_out = attn_out.transpose(1, 2).contiguous()
            attn_out = attn_out.view(*attn_out.shape[:2], -1)
            x = self.self_attn.out_proj(attn_out).transpose(0, 1).contiguous()
            if need_head_weights:
                attn = attn_probs.transpose(0, 1).contiguous()
            else:
                attn = attn_probs.mean(1)
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=True,
                need_head_weights=need_head_weights,
                attn_mask=self_attn_mask,
            )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x, attn


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class ContactPredictionHead(nn.Module):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""

    def __init__(
        self,
        in_features: int,
        prepend_bos: bool,
        append_eos: bool,
        bias=True,
        eos_idx: Optional[int] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        if append_eos and eos_idx is None:
            raise ValueError("Using an alphabet with eos token, but no eos token was passed in.")
        self.eos_idx = eos_idx
        self.regression = nn.Linear(in_features, 1, bias)
        self.activation = nn.Sigmoid()

    def forward(self, tokens, attentions):
        # remove eos token attentions
        if self.append_eos:
            eos_mask = tokens.ne(self.eos_idx).to(attentions)
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
            attentions = attentions * eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]
        # remove cls token attentions
        if self.prepend_bos:
            attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # features: B x C x T x T
        attentions = attentions.to(
            self.regression.weight.device
        )  # attentions always float32, may need to convert to float16
        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)
        return self.activation(self.regression(attentions).squeeze(3))


class MINTContactHead(nn.Module):

    def __init__(self, esm2_output_dim: int):

        super().__init__()

        self.linear_in = nn.Linear(
            esm2_output_dim, esm2_output_dim
        )  # (B x L x L x E) x (E x E) = (B x L x L x E)
        self.head_gelu = nn.GELU()
        self.linear_out = nn.Linear(
            esm2_output_dim, 1
        )  # (B x L x l x E) x (E x 1) = (B x L x L x 1)

    def forward(self, x):
        return self.linear_out(self.head_gelu(self.linear_in(x))).squeeze(-1)  # (B x L x L)
