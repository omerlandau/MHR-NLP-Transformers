# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from fairseq import utils
from torch import Tensor, nn
from torch.nn import Parameter
from fairseq.incremental_decoding_utils import with_incremental_state
import numpy as np
import time
import scipy.spatial as sp

@with_incremental_state
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=False,
            encoder_decoder_attention=False,
            mask_layer=None,
            mask_head=None,
            mask_layer_type=None,
            head_confidence_method=None,

    ):
        super().__init__()
        self.mask_layer = mask_layer
        self.mask_head = mask_head
        self.mask_layer_type = mask_layer_type
        self.head_confidence_method = head_confidence_method
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_conf = ["monkey"]
        self.head_dim = embed_dim // num_heads
        self.bsz = 0
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        if(self_attention):
            print("#########SELF_ATTENTION##########")
        else:
            print("########ENCODER_ATTENTION########")

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        #self.alphas = Parameter(torch.zeros((num_heads, num_heads)))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)


        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

        self.enable_torch_version = False
        # if hasattr(F, "multi_head_attention_forward"):
        #    self.enable_torch_version = True
        # else:
        #    self.enable_torch_version = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
            #self.alphas.data.fill_diagonal_(1)
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
            #self.alphas.data.fill_diagonal_(1)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
            calc_head_importance=False
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """

        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        self.bsz = bsz
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if (
                self.enable_torch_version
                and not self.onnx_trace
                and incremental_state is None
                and not static_kv
                # A workaround for quantization to work. Otherwise JIT compilation
                # treats bias in linear module as method.
                and not torch.jit.is_scripting()
        ):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:

            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )
        q = (
            q.contiguous()
                .view(tgt_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
            )
        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        k0 =k

        q0 = q

        v0 = v

        """
        ##########omertemp##########
        t = k.view(bsz, self.num_heads, -1, self.head_dim)
        g = v.view(bsz, self.num_heads, -1, self.head_dim)
        f = q.view(bsz, self.num_heads, -1, self.head_dim)
        print("KEY0")
        print(t[0, 0, :, :])
        print("VALUE0")
        print(g[0, 0, :, :])
        print("QUERY0")
        print(f[0, 0, :, :])
        print("KEY1")
        print(t[0, 1, :, :])
        print("VALUE1")
        print(g[0, 1, :, :])
        print("QUERY1")
        print(f[0, 1, :, :])
        print("KEY2")
        print(t[0, 2, :, :])
        print("VALUE2")
        print(g[0, 2, :, :])
        print("QUERY2")
        print(f[0, 2, :, :])
        print("KEY3")
        print(t[0, 3, :, :])
        print("VALUE3")
        print(g[0, 3, :, :])
        print("QUERY3")
        print(f[0, 3, :, :])


        print("Weights")

        k = self.k_proj.weight.view(self.num_heads,self.head_dim,embed_dim)

        q = self.q_proj.weight.view(self.num_heads, self.head_dim, embed_dim)

        v = self.v_proj.weight.view(self.num_heads,self.head_dim,embed_dim)

        print("KEY0")
        print(k[0, :, :])
        print("VALUE0")
        print(v[0, :, :])
        print("QUERY0")
        print(q[0,:, :])
        print("KEY1")
        print(k[1, :, :])
        print("VALUE1")
        print(v[1, :, :])
        print("QUERY1")
        print(q[1, :, :])
        print("KEY2")
        print(k[ 2, :, :])
        print("VALUE2")
        print(v[ 2, :, :])
        print("QUERY2")
        print(q[0, :, :])
        print("KEY3")
        print(k[3, :, :])
        print("VALUE3")
        print(v[3, :, :])
        print("QUERY3")
        print(q[3, :, :])
        """

        k = self.k_proj.weight.view(self.num_heads, self.head_dim, embed_dim)

        q = self.q_proj.weight.view(self.num_heads, self.head_dim, embed_dim)

        v = self.v_proj.weight.view(self.num_heads, self.head_dim, embed_dim)


        cosine_sim = (torch.matmul(k[0,:,:].flatten(), k[1,:,:].flatten()) / (torch.norm(k[0,:,:].flatten()) * torch.norm(k[1,:,:].flatten())))

        print("KEY0-1sim")

        print(cosine_sim)

        print(cosine_sim.shape)

        print("KEY0-7sim")

        cosine_sim = (torch.matmul(k[0,:,:].flatten(), k[7,:,:].flatten()) / (torch.norm(k[0,:,:].flatten()) * torch.norm(k[2,:,:].flatten())))

        print(cosine_sim)

        print("KEY0-3sim")

        cosine_sim = (torch.matmul(k[0, :, :].flatten(), k[3, :, :].flatten()) / (torch.norm(k[0, :, :].flatten()) * torch.norm(k[3, :, :].flatten())))

        print(cosine_sim)

        print("VALUE0-1sim")

        cosine_sim = (torch.matmul(v[0, :, :].flatten(), v[1, :, :].flatten()) / (torch.norm(v[0, :, :].flatten()) * torch.norm(v[1, :, :].flatten())))

        print(cosine_sim)

        print("VALUE0-2sim")

        cosine_sim = (torch.matmul(v[0, :, :].flatten(), v[2, :, :].flatten()) / (torch.norm(v[0, :, :].flatten()) * torch.norm(v[2, :, :].flatten())))

        print(cosine_sim)

        print("VALUE0-3sim")

        cosine_sim = (torch.matmul(v[0, :, :].flatten(), v[3, :, :].flatten()) / (torch.norm(v[0, :, :].flatten()) * torch.norm(v[3, :, :].flatten())))

        print(cosine_sim)

        print("QUERY0-1sim")

        cosine_sim = (torch.matmul(q[0, :, :].flatten(), q[0, :, :].flatten()) / (torch.norm(q[0, :, :].flatten()) * torch.norm(q[0, :, :].flatten())))

        print(cosine_sim)

        print("QUERY0-2sim")

        cosine_sim = (torch.matmul(q[0, :, :].flatten(), q[2, :, :].flatten()) / (torch.norm(q[0, :, :].flatten()) * torch.norm(q[2, :, :].flatten())))

        print(cosine_sim)

        print("QUERY5-3sim")

        cosine_sim = (torch.matmul(q[5, :, :].flatten(), q[3, :, :].flatten()) / (torch.norm(q[5, :, :].flatten()) * torch.norm(q[3, :, :].flatten())))

        print(cosine_sim)

        def pairwise_distances(x, y=None):
            '''
            Input: x is a Nxd matrix
                   y is an optional Mxd matirx
            Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                    if y is not given then use 'y=x'.
            i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
            '''
            x_norm = (x ** 2).sum(1).view(-1, 1)
            if y is not None:
                y_norm = (y ** 2).sum(1).view(1, -1)
            else:
                y = x
                y_norm = x_norm.view(1, -1)

            dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
            return dist

        print("KEY0-1Pairwisedist")

        cosine_sim = torch.norm(k[0, :, :]-k[1, :, :])

        print(cosine_sim)

        print("KEY0-2sim")

        cosine_sim = torch.norm(k[0, :, :]- k[2, :, :])

        print(cosine_sim)

        print("KEY0-3sim")

        cosine_sim = torch.norm(k[0, :, :]- k[3, :, :])

        print(cosine_sim)

        print("VALUE0-1sim")

        cosine_sim = torch.norm(v[0, :, :]- v[1, :, :])

        print(cosine_sim)

        print("VALUE0-2sim")

        cosine_sim = torch.norm(v[0, :, :]- v[2, :, :])

        print(cosine_sim)

        print("VALUE0-3sim")

        cosine_sim = torch.norm(v[0, :, :]-v[3, :, :])

        print(cosine_sim)

        print("QUERY0-1sim")

        cosine_sim = torch.norm(q[0, :, :]-q[1, :, :])

        print(cosine_sim)

        print("QUERY0-2sim")

        cosine_sim = torch.norm(q[0, :, :]- q[2, :, :])

        print(cosine_sim)

        print("QUERY5-3sim")

        cosine_sim = torch.norm(q[5, :, :]-q[3, :, :])

        print(cosine_sim)


        k= k0

        q = q0

        v= v0



        #####done temp######

        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = MultiheadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )

        if self.mask_head is not None:
            head_masking_vector = torch.ones(self.num_heads)
            head_masking_vector[self.mask_head] = 0
            head_masking_vector = head_masking_vector.view(1, self.num_heads, 1, 1).to(attn_weights_float.device)
            attn_weights_float = attn_weights_float.view(self.num_heads, bsz, tgt_len, src_len)

            attn_weights_float = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len) * head_masking_vector
            attn_weights_float = attn_weights_float.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_float.type_as(attn_weights)
        conf = None

        a = attn_weights.clone().view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)


        ## computing confidence of all heads over bsz sentences

        ## heads is an np array of shape [head_nums+1] which contains confidence*bsz for each head and bsz:
        ## [conf_h_1*bsz,conf_h_2*bsz,...,conf_h_n*bsz,bsz]
        # Viota's confidence is based on:
        # Word attn confidence is an upgraded more delicate version of conf,
        # where
        if self.head_confidence_method is not None:

            if attn_weights is not None:
                if self.head_confidence_method == "base":
                    a = attn_weights.clone().view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
                    a[:, :, -1, -1] = torch.zeros((self.num_heads, bsz))
                    heads = a[:, :, :, :].max(dim=3)
                    heads = heads[0].max(dim=2)
                    heads = heads[0].sum(dim=1) / bsz
                else:
                    a = attn_weights.clone().view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
                    a[:, :, -1, -1] = torch.zeros((self.num_heads, bsz))
                    heads = a[:, :, :, :].max(dim=2)
                    heads = heads[0].sum(dim=2) / (src_len - 1)
                    heads = heads.sum(dim=1) / bsz
                    heads = heads


            # Take max for each source word, than average all
            # for j in range(self.num_heads):
            #    conf_temp = 0
            #    for batch in range(bsz):
            #        word_attn_sum = 0
            #        for tgt in range(tgt_len - 1):
            #            word_attn_sum += attn_weights.view(self.num_heads, bsz, tgt_len, src_len)[j, batch, tgt,
            #                             :-1].max()
            #        conf_temp += word_attn_sum / (tgt_len - 1)
            #    word_max["heads"].append(conf_temp)
            conf = heads

        self.head_conf = conf

        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )

        #print(self.alphas)

        assert v is not None

        ctx = torch.bmm(attn_probs, v)  # Thats what I called 'Z' in my summary.
        save_ctx = ctx.view(bsz, self.num_heads, tgt_len, self.head_dim)
        ctx = save_ctx.view(bsz * self.num_heads, tgt_len, self.head_dim)

        #z = ctx.contiguous().view(bsz, self.num_heads,tgt_len,self.head_dim).transpose(0,1)

        #b = z.contiguous().view(self.num_heads, tgt_len*bsz*self.head_dim)

        #self.alphas.requires_grad = True

        #b = torch.mm(self.alphas,b)

        #ctx = b.contiguous().view(self.num_heads, bsz,tgt_len,self.head_dim).transpose(0,1)

        #ctx = ctx.contiguous().view(bsz * self.num_heads, tgt_len, self.head_dim)






        assert list(ctx.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and ctx.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            ctx = ctx.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            ctx = ctx.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(ctx)
        attn_weights: Optional[Tensor] = None
        if calc_head_importance:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            if need_weights:
                attn_weights = attn_weights_float.view(
                    bsz, self.num_heads, tgt_len, src_len
                ).transpose(1, 0)
                if not need_head_weights:
                    # average attention weights over heads
                    attn_weights = attn_weights.mean(dim=0)
        return attn, attn_weights, save_ctx

    @staticmethod
    def _append_prev_key_padding_mask(
            key_padding_mask: Optional[Tensor],
            prev_key_padding_mask: Optional[Tensor],
            batch_size: int,
            src_len: int,
            static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
            self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(0) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
            self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim: 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim:]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                                                           dim: 2 * dim
                                                           ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim:]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value
