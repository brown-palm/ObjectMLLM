# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from torch.nn import Embedding, Linear
import torch
from .tokenizer import Tokenizer_llama3

def set_precision(t, precision):
    if precision == 'bf16':
        return t.bfloat16()
    elif precision == 'fp16':
        return t.half()
    elif precision == 'fp32':
        return t.float()
    else:
        raise ValueError(f"Unknown half format {precision}")


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048
    adapter_len: int=10
    adapter_layer: int=30
    precision: str='bf16'

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.max_feats = args.max_feats
        self.precision = args.precision

        self.wq = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)).cuda()
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim)).cuda()
        self.gate = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, video_start=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        if adapter is not None:
            adapter_len = adapter.shape[1]
            adapter_k = self.wk(adapter).view(1, adapter_len, self.n_local_kv_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_v = self.wv(adapter).view(1, adapter_len, self.n_local_kv_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            xk = torch.cat([adapter_k, xk], dim=1)
            xv = torch.cat([adapter_v, xv], dim=1)
            extra_mask = torch.zeros(1, 1, seqlen, adapter_len).to(mask)
            mask = torch.cat([extra_mask, mask], dim=-1)
        keys = xk
        values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        if adapter is not None:            
            adapter_scores = (F.softmax(scores[..., :adapter_len].float(), dim=-1) * self.gate.tanh()).type_as(xq)
            vt_scores = scores[..., adapter_len:].clone()
            vt_scores = F.softmax(vt_scores.float(), dim=-1).type_as(xq)
            scores = torch.cat([adapter_scores, vt_scores], dim=-1)
        else:
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(dim, hidden_dim, bias=False)
        self.w2 = Linear(hidden_dim, dim, bias=False)
        self.w3 = Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of, ffn_dim_multiplier=args.ffn_dim_multiplier)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, video_start=None):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter, video_start)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, args):
        super().__init__()
        params.max_feats = args.max_feats
        params.bias = args.bias
        self.args = args
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.max_feats = args.max_feats
        self.precision = args.precision
        dim_dict = {'clip': 768, 'dinov2': 1024, 'siglip':1152}
        self.input_dims = args.input_dims

        self.mm_used = args.mm_used
        
        self.tok_embeddings = Embedding(params.vocab_size, params.dim)

        self.adapter_query = Embedding(params.adapter_len * params.adapter_layer, params.dim)

        if args.proj_type == 'linear':
            self.mm_projs = nn.ModuleDict({
                m: Linear(self.input_dims[m], params.dim * args.dim_expand, bias=False)
                for m in self.mm_used
            })
        elif 'mlp' in args.proj_type:
            n_layers = int(args.proj_type.split('_')[-1])
            d = {}
            for m in self.mm_used:
                modules = [nn.Linear(self.input_dims[m], params.dim, bias=False)]
                for _ in range(1, n_layers):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(params.dim, params.dim * (args.dim_expand if _ == n_layers - 1 else 1), bias=False))
                d[m] = nn.Sequential(*modules)
            self.mm_projs = nn.ModuleDict(d)
        else:
            raise ValueError(f"Unknown projection type {args.proj_type}")

        if args.zero_init:
            for m in self.mm_used:
                if args.proj_type == 'linear':
                    nn.init.constant_(self.mm_projs[m].weight, 0.0)
                else:
                    # Only zero initialize the last linear layer
                    nn.init.constant_(self.mm_projs[m][-1].weight, 0.0)

        self.adapter_len = params.adapter_len
        self.adapter_layer = params.adapter_layer

        self.vqa_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.inference_criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2, self.params.rope_theta)

        self.tokenizer = Tokenizer_llama3(model_path=f'{args.llama_model_path}/tokenizer.model')

    def forward(self, data, use_vis, inference=False):
        mm_features = data['mm_features']

        if inference and data['text_id']['vqa'].shape[1] > 5:
            # Special mini-batch processing for the Babel dataset
            assert data['text_id']['vqa'].shape[1] == 50
            from copy import deepcopy
            outputs = []
            with torch.no_grad():
                for i in range(0, 50, 5):
                    new_data = deepcopy(data)
                    new_data['text_id']['vqa'] = new_data['text_id']['vqa'][:, i: i + 5]
                    new_data['label']['vqa'] = new_data['label']['vqa'][:, i: i + 5]
                    mini_output, d = self(new_data, use_vis, inference=True)
                    outputs.append(mini_output)
            logits = torch.cat(outputs, dim=1)
            logits = logits.reshape(1, 50, -1)
            return logits, {'fnorm': None}

        vqa_id = data['text_id']['vqa'].cuda()
        vqa_label = data['label']['vqa'].cuda()
        mm_starts = data['mm_starts']['vqa']
        
        bsz, n_options, seqlen = vqa_id.shape
        vqa_id = vqa_id.reshape(-1, seqlen)
        vqa_label = vqa_label.reshape(-1, seqlen)
        vqa_label = vqa_label[:, 1:].flatten()
                
        with torch.no_grad():
            vqa_h = self.tok_embeddings(vqa_id)
            
        freqs_cis = self.freqs_cis.to(vqa_h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=vqa_h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(vqa_h)
        start_pos = 0
        
        adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, self.params.dim).unsqueeze(1)

        # Process the multimodal feature. Only useful when using the embedding projector
        fnorm = None
        mm_features_proj = [{} for _ in range(bsz)]
        for m in self.mm_used:
            feats = []

            # Concatenate all the multimodal feature for modal m
            for sample in mm_features:
                if len(sample[m]) > 0:
                    feats.append(torch.cat(sample[m], dim=0).cuda())
                else:
                    feats.append(torch.Tensor([]).cuda())
            split_sizes_ = [x.shape[0] for x in feats]
            feats = torch.cat(feats, dim=0)
            if feats.shape[0] == 0:
                continue

            # Apply the embedding projector
            feats_proj = self.mm_projs[m](feats) # n x (dim * expand)
            n, d = feats_proj.shape
            assert d % self.params.dim == 0
            feats_proj = feats_proj.reshape(n * d // self.params.dim, self.params.dim)
            feats_proj = set_precision(feats_proj, self.precision)
            fnorm = feats_proj.norm(dim=-1).mean()

            # Split the multimodal feature back to the original shape
            assert feats_proj.shape[0] % sum(split_sizes_) == 0
            multiplier = feats_proj.shape[0] // sum(split_sizes_)
            split_sizes = [z * multiplier for z in split_sizes_]
            feats_proj = torch.split(feats_proj, split_sizes, dim=0)
            for i in range(bsz):
                mm_features_proj[i][m] = torch.split(feats_proj[i], self.max_feats[m], dim=0)
        
        vqa_h = vqa_h.clone().reshape(bsz, n_options, seqlen, self.params.dim)
        for i in range(bsz):
            cnt = {m: 0 for m in self.mm_used}
            for m, st in mm_starts[i]:
                feats = mm_features_proj[i][m][cnt[m]] # max_feats x n_dim
                cnt[m] += 1
                feats = feats.unsqueeze(0).repeat(n_options, 1 ,1) # n_options x max_feats x n_dim
                vqa_h[i, :, st: st + self.max_feats[m]] = feats
        vqa_h = vqa_h.reshape(-1, seqlen, self.params.dim)
        
        for i, layer in enumerate(self.layers[-1 * self.adapter_layer:]):
            vqa_h = layer(vqa_h, start_pos, freqs_cis, mask, set_precision(adapter[i], self.precision)) 
        
        vqa_h = self.norm(vqa_h)
        vqa_output = self.output(vqa_h)
        vqa_output = vqa_output[:, :-1, :].reshape(-1, self.vocab_size)
        vqa_loss = self.vqa_criterion(vqa_output, vqa_label)
        
        if inference:
            logits = self.inference_criterion(vqa_output, vqa_label)
            logits = logits.reshape(bsz, n_options, -1)
            return logits, {'fnorm': fnorm}
        else:
            return vqa_loss