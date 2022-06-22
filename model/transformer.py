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
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(size)
        self.size = size

    def forward(self, x, q, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.layer_norm(x)
        x = x + self.dropout(self.cross_attn(q, x, x, mask))

        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer1, layer2, N):
        super(Decoder, self).__init__()
        self.layers1 = clones(layer1, N)
        self.layers2 = clones(layer2, N)
        self.norm = nn.LayerNorm(layer1.size)

    def forward(self, x, q, mask):
        for layer1, layer2 in zip(self.layers1, self.layers2):
            x = layer1(x, q, mask)
            x = layer2(x, mask)
        return self.norm(x)


class TransformerRescorer(nn.Module):
    """Main Transformer rescorer model"""

    def __init__(self, device, pretrain_embed, checkpoint_path, num_words, opt, pretrain_model):
        super(TransformerRescorer, self).__init__()

        self.hidden_dim = opt.hidden_dim
        self.attn_head = opt.attn_head
        self.bert_dim = opt.bert_dim
        self.num_words = num_words
        self.device = device

        ### embedding layer
        self.text_encoder = pretrain_model
        for param in self.text_encoder.parameters():
            param.require_grads = False

        self.audio_encoder = nn.Linear(768, self.bert_dim)
        self.fc = nn.Linear(self.bert_dim, self.num_words)


        ### Attention layer
        self.attn = MultiHeadAttention(self.attn_head, self.bert_dim, dropout=0.)
        self.attn2 = MultiHeadAttention(self.attn_head, self.bert_dim, dropout=0.)
        self.add_pe = PositionalEncoding(self.bert_dim, 0.)

        ### rescorer
        self.rescorer = Decoder(DecoderLayer(self.bert_dim,
                                        MultiHeadAttention(self.attn_head, self.bert_dim, dropout=0.),
                                        MultiHeadAttention(self.attn_head, self.bert_dim, dropout=0.),
                                        PositionwiseFeedForward(self.bert_dim, self.hidden_dim, 0.),
                                        0.1),
                                SelfDecoderLayer(self.bert_dim,
                                            MultiHeadAttention(self.attn_head, self.bert_dim, dropout=0.),
                                            PositionwiseFeedForward(self.bert_dim, self.hidden_dim, 0.),
                                            0.1),
                                            N=2)
        self.criterion = nn.CrossEntropyLoss(reduce=False).to(self.device)
        self.criterion_ce = nn.CrossEntropyLoss().to(self.device)

    def cross_entropy_loss(self, logits, batch):
        """Compute cross entropy loss"""

        (result_ids, result_token_masks, result_masks, lengths, result_audio, result_wers, truth_ids,
         truth_token_masks, result_asr_features) = batch
        truth_ids = truth_ids.to(self.device)  # (b, t)
        truth_token_masks = truth_token_masks.to(self.device)  # (b, t)

        b, t = truth_ids.shape

        targets = Variable(torch.zeros(b, t)).to(self.device)  # (b, d, t)
        targets[:, :-1] = truth_ids[:, 1:]

        loss = self.criterion_ce(logits.reshape(-1, self.num_words), targets.reshape(-1).long())
        return loss


    def mwer_loss(self, logits, batch):
        """Compute minimum word error rate weighted cross entropy loss"""

        (result_ids, result_token_masks, result_masks, lengths, result_audio, result_wers, truth_ids, truth_token_masks, result_asr_features) = batch
        result_ids = result_ids.to(self.device)  # (b, d, t)
        result_masks = result_masks.to(self.device)  # (b, d)
        lengths = lengths.to(self.device)  # (b,)
        result_wers = result_wers.to(self.device)  # (b, d)

        b, d, t = result_ids.shape

        targets = Variable(torch.zeros(b, d, t)).to(self.device)  # (b, d, t)
        targets[:, :, :-1] = result_ids[:, :, 1:]

        logits = logits.view(*result_ids.shape, self.num_words) # (b, d, t, self.bert_dim)

        all_loss = 0

        # Uncomment the following line for raw MWER calculation
        # for i in range(b):
        #     ls = logits[i, :lengths[i], :, :] # (d,t,h)
        #     ts = targets[i, :lengths[i], :].unsqueeze(2).long() # (d,t,1)
        #
        #     probs = nn.Softmax(dim=2)(ls)
        #     probs_for_target = probs.gather(2, ts).squeeze(2) # (d,t)
        #     probs_for_target = torch.sum(torch.log(probs_for_target), dim=1) # (d,)
        #     probs_for_target = probs_for_target / (probs_for_target.sum()+1e-4)
        #     wer_average = torch.mean(result_wers[i, :lengths[i]])
        #     per_sample_loss = torch.sum(probs_for_target * (result_wers[i, :lengths[i]] - wer_average))
        #     all_loss += per_sample_loss

        for i in range(b):
            ls = logits[i, :lengths[i], :, :].reshape(-1, self.num_words)
            ts = targets[i, :lengths[i], :].reshape(-1)
            losses = self.criterion(ls, ts.long()) # (num_of_nbest * t,)
            losses = losses.view(lengths[i], t).sum(dim=1) # (num_of_nbest,)
            losses = nn.Softmax()(losses)
            wer_average = torch.mean(result_wers[i, :lengths[i]])
            per_sample_loss = -torch.sum(losses * (result_wers[i, :lengths[i]] - wer_average))
            all_loss += per_sample_loss

        return all_loss

    def ce_and_mwer_loss(self, logits, batch):
        """Compute minimum word error rate weighted cross entropy loss"""

        (result_ids, result_token_masks, result_masks, lengths, result_audio, result_wers, truth_ids, truth_token_masks, result_asr_features) = batch
        result_ids = result_ids.to(self.device)  # (b, d, t)
        result_masks = result_masks.to(self.device)  # (b, d)
        lengths = lengths.to(self.device)  # (b,)
        result_wers = result_wers.to(self.device)  # (b, d)

        b, d, t = result_ids.shape

        targets = Variable(torch.zeros(b, d, t)).to(self.device)  # (b, d, t)
        targets[:, :, :-1] = result_ids[:, :, 1:]

        logits = logits.view(*result_ids.shape, self.num_words) # (b, d, t, self.bert_dim)

        mwer_loss = 0
        for i in range(b):
            ls = logits[i, :lengths[i], :, :].reshape(-1, self.num_words)
            ts = targets[i, :lengths[i], :].reshape(-1)
            losses = self.criterion(ls, ts.long()) # (num_of_nbest * t,)
            losses = losses.view(lengths[i], t).sum(dim=1) # (num_of_nbest,)
            losses = nn.Softmax()(losses)
            wer_average = torch.mean(result_wers[i, :lengths[i]])
            per_sample_loss = -torch.sum(losses * (result_wers[i, :lengths[i]] - wer_average))
            mwer_loss += per_sample_loss

        ce_loss = self.criterion_ce(logits.reshape(-1, self.num_words), targets.reshape(-1).long())

        return mwer_loss + 0.1*ce_loss


    def forward(self, batch):
        """forward function"""

        (result_ids, result_token_masks, result_masks, lengths, result_audio, result_wers, truth_ids, truth_token_masks, result_asr_features) = batch
        result_ids = result_ids.to(self.device) # (b, d, t)
        result_token_masks = result_token_masks.to(self.device) # (b, d, t)
        result_audio = result_audio.to(self.device) # (b, h_a)
        truth_ids = truth_ids.to(self.device)  # (b, t)
        truth_token_masks = truth_token_masks.to(self.device)  # (b, t)

        # Uncomment the following line for truth and audio data
        # b, t = truth_ids.shape
        # outputs = self.text_encoder(truth_ids, truth_token_masks)
        # text_embeddings = outputs.hidden_states[-1]
        # audio_embeddings = self.audio_encoder(result_audio)
        #
        # attention_mask = torch.Tensor(b, t, t).byte().to(self.device)
        # for i in range(b):
        #     padding_utter = (truth_token_masks[i, :].sum(-1) != 0)
        #     attention_mask[i] = padding_utter.unsqueeze(0).repeat(t, 1) & subsequent_mask(t).to(self.device)

        # for hypothesis and audio data
        b, d, t = result_ids.shape

        # encode
        result_ids = result_ids.view(-1, t) # (bxd, h)
        result_token_masks = result_token_masks.view(-1, t)
        outputs = self.text_encoder(result_ids, result_token_masks)
        text_embeddings = outputs.hidden_states[-1] # (bxd, t, h)

        audio_embeddings = self.audio_encoder(result_audio).unsqueeze(1).repeat(d, 1, 1)  # (b*d, 1, h_a)

        # NBT
        attention_mask = torch.Tensor(b*d, t, t).byte().to(self.device)
        for i in range(b*d):
            padding_utter = (result_token_masks[i, :].sum(-1) != 0)
            attention_mask[i] = padding_utter.unsqueeze(0).repeat(t, 1) & subsequent_mask(t).to(self.device)

        hidden = self.rescorer(text_embeddings, audio_embeddings, attention_mask) # (b*d, t, h)
        logits = self.fc(hidden)

        return logits
