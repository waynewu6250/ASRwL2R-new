import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, AutoModel, BertConfig, AdamW, BertForPreTraining, BertForMaskedLM, AutoConfig, AutoModelForMaskedLM
from model.bert_model import BertEmbedding
from torch.autograd import Variable
from torch.nn.init import xavier_normal_ as nr_init
from model.ptranking.metric.metric_utils import get_delta_ndcg
import os

class BertLTRRescorer(nn.Module):
    """Main Bert rescorer model"""

    def __init__(self, device, pretrain_embed, checkpoint_path, num_words, opt, pretrain_model):
        super(BertLTRRescorer, self).__init__()

        self.hidden_dim = opt.hidden_dim
        self.attn_head = opt.attn_head
        self.bert_dim = opt.bert_dim
        self.num_words = num_words
        self.gpu = torch.cuda.is_available()
        self.device = device
        self.sigma = 1.0

        ### Uncomment the following lines for point-wise ltr model (embedding layer)
        # self.bert = pretrain_model
        # self.num_features = 14#+768*2
        # self.sf = self.config_neural_scoring_function()
        # self.fc = nn.Linear(self.bert_dim, 1)

        # Add audio & text
        self.bert = pretrain_model

        self.audio_encoder = nn.Linear(768, self.bert_dim)
        self.fc_truth = nn.Linear(self.bert_dim, 1)
        self.fc = nn.Linear(self.bert_dim * 2 + 14, 1)

        # listwise setting
        self.rnn = nn.LSTM(input_size=self.bert_dim + 14,
                           hidden_size=self.bert_dim,
                           batch_first=True,
                           bidirectional=True,
                           num_layers=1)
        self.classifier_rnn = nn.Linear(self.bert_dim * 2, 1)
        self.dropout = nn.Dropout(0.1)


    def forward(self, batch):
        """forward function"""

        (result_ids, result_token_masks, result_masks, lengths, result_audio, result_ranks, truth_ids,
         truth_token_masks, result_asr_features) = self.get_batch(batch)

        # # 1. Point-wise: Only sparse feature
        # batch_output = self.sf(result_asr_features)
        # batch_preds = torch.squeeze(batch_output, dim=2)

        # 2. Point-wise: sparse feature, text and audio
        # b, d, t = result_ids.shape
        # result_ids = result_ids.view(-1, t)  # (bxd, h)
        # result_token_masks = result_token_masks.view(-1, t)
        # outputs = self.bert(result_ids, result_token_masks)
        # text_embeddings = outputs.hidden_states[-1]  # (bxd, t, h)
        # audio_embeddings = self.audio_encoder(result_audio).unsqueeze(1).repeat(1, d, 1)  # (b, d, h_a)
        #
        # # generate scores
        # head = text_embeddings[:, 0, :] # (bxd, h)
        # head = head.view(b,d,-1)
        # # logits = self.fc(head) # (bxd, 1)
        # fused = torch.cat((audio_embeddings, head, result_asr_features), dim=2)
        #
        # batch_output = self.sf(fused)
        # batch_preds = torch.squeeze(batch_output, dim=2)

        ################ list-wise ltr ################
        # for hypothesis and audio data
        b, d, t = result_ids.shape

        # hypothesis
        result_ids = result_ids.view(-1, t)  # (bxd, h)
        result_token_masks = result_token_masks.view(-1, t)
        outputs = self.bert(result_ids, result_token_masks)
        text_embeddings = outputs.hidden_states[-1]  # (bxd, t, h)
        head = text_embeddings[:, 0, :]  # (bxd, h)

        # audio
        audio_embeddings = self.audio_encoder(result_audio).repeat(d, 1)  # (b*d, h_a)
        result_asr_features = result_asr_features.view(b * d, -1)

        # rewrites
        b, t = truth_ids.shape
        outputs = self.bert(truth_ids, truth_token_masks)
        truth_embeddings = outputs.hidden_states[-1]  # (b, t, h)
        truth_head = truth_embeddings[:, 0, :]  # (b, h)
        truth_head_repeat = truth_head.repeat(d, 1)  # (b*d, h)

        # list-wise setting
        audio_hidden = self.audio_encoder(result_audio).unsqueeze(0).repeat(2, 1, 1)  # (2, b, h_a)
        inputs = torch.cat((head, result_asr_features), dim=1)
        inputs = inputs.view(b, d, -1)
        (h_0, c_0) = audio_hidden, torch.zeros(*audio_hidden.shape).to(self.device)

        packed = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths.cpu().numpy(), batch_first=True,
                                                         enforce_sorted=False)
        rnn_out, _ = self.rnn(packed)  # (h_0, c_0))
        seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        rnn_out = self.dropout(seq_unpacked)
        logits = self.classifier_rnn(rnn_out)  # (b,d,1)
        logits = torch.squeeze(logits, dim=2)

        return logits

    def learning_to_rank_loss(self, batch_preds, batch):
        """Learning to rank loss"""
        (result_ids, result_token_masks, result_masks, lengths, result_audio, result_ranks, truth_ids,
         truth_token_masks, result_asr_features) = self.get_batch(batch)

        b, d, t = result_ids.shape
        batch_preds_clean = []
        batch_loss = 0
        total_hyps = 0
        for i in range(b):
            total_hyps += lengths[i]
            scores = batch_preds[i][:lengths[i]]
            ranks = result_ranks[i][:lengths[i]]

            batch_loss += self.point_loss(scores.unsqueeze(0), ranks.unsqueeze(0))
            batch_preds_clean += scores.detach().cpu().numpy().tolist()
        return batch_loss, total_hyps, batch_preds_clean

    def point_loss(self, batch_preds, batch_stds):
        """ Main point loss for learning to rank
        :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
        :param batch_stds:  [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
        """
        batch_preds_sorted, batch_preds_sorted_inds = torch.sort(batch_preds, dim=1, descending=True)  # sort documents according to the predicted relevance
        batch_stds_sorted_via_preds = torch.gather(batch_stds, dim=1, index=batch_preds_sorted_inds)  # reorder batch_stds correspondingly so as to make it consistent. BTW, batch_stds[batch_preds_sorted_inds] only works with 1-D tensor

        batch_std_diffs = torch.unsqueeze(batch_stds_sorted_via_preds, dim=2) - torch.unsqueeze(batch_stds_sorted_via_preds, dim=1)  # standard pairwise differences, i.e., S_{ij}
        batch_std_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)  # ensuring S_{ij} \in {-1, 0, 1}
        batch_std_p_ij = 0.5 * (1.0 + batch_std_Sij)

        batch_s_ij = torch.unsqueeze(batch_preds_sorted, dim=2) - torch.unsqueeze(batch_preds_sorted, dim=1)  # computing pairwise differences, i.e., s_i - s_j
        batch_p_ij = 1.0 / (torch.exp(-self.sigma * batch_s_ij) + 1.0)

        batch_delta_ndcg = get_delta_ndcg(batch_ideally_sorted_stds=batch_stds, batch_stds_sorted_via_preds=batch_stds_sorted_via_preds, gpu=self.gpu)

        # about reduction, mean leads to poor performance, a probable reason is that the small values due to * lambda_weight * mean
        b_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                            target=torch.triu(batch_std_p_ij, diagonal=1),
                                            weight=torch.triu(batch_delta_ndcg, diagonal=1), reduction='sum')

        return b_loss

    def get_batch(self, batch):
        """Get batch to correct device"""

        new_batch = []
        for data in batch:
            new_batch.append(data.to(self.device))
        return tuple(new_batch)