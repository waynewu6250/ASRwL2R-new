"""
Copyright <first-edit-year> Amazon.com, Inc. and its affiliates. All Rights Reserved.
 
SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0

Licensed under the Amazon Software License (the "License").
You may not use this file except in compliance with the License.
A copy of the License is located at

  http://aws.amazon.com/asl/

or in the "license" file accompanying this file. This file is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
express or implied. See the License for the specific language governing
permissions and limitations under the License.
"""
import os.path
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn import CrossEntropyLoss
from torch.nn import CosineEmbeddingLoss
from model.bert_model import BertEmbedding


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def subsequent_mask(size):
    """""Mask out subsequent positions."""

    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

###################################################


class MultiHeadAttention(nn.Module):
    """Main Multi-head attention sublayer"""
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores

def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SelfDecoderLayer(nn.Module):
    """Decoder is made up of self-attn and feed forward (defined below)"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(SelfDecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    """Decoder is made up of self-attn, cross-attn and feed forward (defined below)"""
    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(size)
        self.size = size

    def forward(self, x, q, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = x + self.dropout(self.cross_attn(q, x, x, mask))
        x = self.layer_norm(x)

        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer1, N):
        super(Decoder, self).__init__()
        self.layers1 = clones(layer1, N)
        # self.layers2 = clones(layer2, N)
        self.norm = nn.LayerNorm(layer1.size)

    def forward(self, x, q, mask):
        # for layer1, layer2 in zip(self.layers1, self.layers2):
        #     x = layer1(x, q, mask)
        #     x = layer2(x, mask)
        for layer1 in self.layers1:
            x = layer1(x, q, mask)
        return self.norm(x)

class TransformerFuser(nn.Module):
    """Main Transformer rescorer model"""

    def __init__(self, device, opt):
        super(TransformerFuser, self).__init__()

        self.hidden_dim = opt.hidden_dim
        self.attn_head = opt.attn_head
        self.bert_dim = opt.bert_dim
        self.device = device

        self.feature_encoder = nn.Linear(self.bert_dim+14, self.bert_dim)
        self.audio_encoder = nn.Linear(self.bert_dim, self.bert_dim)

        ### Attention layer
        self.attn = MultiHeadAttention(self.attn_head, self.bert_dim, dropout=0.)
        self.attn2 = MultiHeadAttention(self.attn_head, self.bert_dim, dropout=0.)
        self.add_pe = PositionalEncoding(self.bert_dim, 0.)

        ### fuser
        self.fuser = Decoder(DecoderLayer(self.bert_dim,
                                        MultiHeadAttention(self.attn_head, self.bert_dim, dropout=0.),
                                        MultiHeadAttention(self.attn_head, self.bert_dim, dropout=0.),
                                        PositionwiseFeedForward(self.bert_dim, self.hidden_dim, 0.),
                                        0.1),
                             # SelfDecoderLayer(self.bert_dim,
                             #            MultiHeadAttention(self.attn_head, self.bert_dim, dropout=0.),
                             #            PositionwiseFeedForward(self.bert_dim, self.hidden_dim, 0.),
                             #            0.1),
                             N=2)


    def forward(self, inputs, result_masks, result_audio):
        """forward function
        :param inputs:          main input hypothesis features (b, d, h)
        :param result_masks:    mask of which location each utterance has hypotheses (b, d)
        :param result_audio:    audio inputs (b, ha)
        """

        b, d = result_masks.shape
        inputs = self.feature_encoder(inputs)
        result_audio = self.audio_encoder(result_audio).unsqueeze(1)

        # NBT
        attention_mask = torch.Tensor(b, d, d).byte().to(self.device)
        for i in range(b):
            padding_utter = (result_masks[i, :].sum(-1) != 0)
            attention_mask[i] = padding_utter.unsqueeze(0).repeat(d, 1) & subsequent_mask(d).to(self.device)

        hidden = self.fuser(inputs, result_audio, attention_mask) # (b*d, t, h)

        return hidden
