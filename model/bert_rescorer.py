import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, AutoModel, BertConfig, AdamW, BertForPreTraining, BertForMaskedLM, AutoConfig, AutoModelForMaskedLM
from model.bert_model import BertEmbedding
from torch.autograd import Variable
import os
from model.transformer_fuser import TransformerFuser

class BertRescorer(nn.Module):
    """Main Bert rescorer model"""

    def __init__(self, device, pretrain_embed, checkpoint_path, num_words, opt, pretrain_model, loss_type):
        super(BertRescorer, self).__init__()

        self.hidden_dim = opt.hidden_dim
        self.attn_head = opt.attn_head
        self.bert_dim = opt.bert_dim
        self.num_words = num_words
        self.device = device
        self.loss_type = loss_type

        ### embedding layer
        self.bert = pretrain_model

        self.audio_encoder = nn.Linear(768, self.bert_dim)
        self.fc_truth = nn.Linear(self.bert_dim, 1)
        self.fc = nn.Linear(self.bert_dim+2, 1)

        # listwise setting
        self.rnn = nn.LSTM(input_size=self.bert_dim+2,
                           hidden_size=self.bert_dim,
                           batch_first=True,
                           bidirectional=True,
                           num_layers=1)
        self.classifier_rnn = nn.Linear(self.bert_dim*2, 1)
        self.dropout = nn.Dropout(0.1)

        # Uncomment the following line for listwise transformer setting
        # self.transformer_fuser = TransformerFuser(self.device, opt)
        # self.classifier_trans = nn.Linear(self.bert_dim, 1)

        # Uncomment the following line for listwise attention setting
        # self.decoder = AttnDecoderRNN(self.bert_dim)
        # self.classifier_rnn_att = nn.Linear(self.bert_dim, 1)

        # loss
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss(reduction='sum')
        self.kl = nn.KLDivLoss(size_average=False, reduce=True)
        self.margin_loss = nn.MarginRankingLoss(margin=5, reduction='sum')
        self.sig = nn.Sigmoid()


    def forward(self, batch):
        """forward function"""

        (result_ids, result_token_masks, result_masks, lengths, result_audio, result_wers, truth_ids,
         truth_token_masks, result_asr_features) = self.get_batch(batch)

        if self.loss_type == 'margin':
            b, t = truth_ids.shape
            outputs = self.bert(truth_ids, truth_token_masks)
            truth_embeddings = outputs.hidden_states[-1] # (b, t, h)
            truth_head = truth_embeddings[:, 0, :]  # (b, h)
            truth_logits = self.fc_truth(truth_head) # (b, 1)

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
        result_asr_features = result_asr_features.view(b*d, -1)

        # rewrites
        b, t = truth_ids.shape
        outputs = self.bert(truth_ids, truth_token_masks)
        truth_embeddings = outputs.hidden_states[-1]  # (b, t, h)
        truth_head = truth_embeddings[:, 0, :]  # (b, h)
        truth_head_repeat = truth_head.repeat(d, 1) # (b*d, h)

        # point-wise setting
        # fused = torch.cat((head, result_asr_features), dim=1)
        # logits = self.fc(fused) # (bxd, 1)

        # list-wise setting
        audio_hidden = self.audio_encoder(result_audio).unsqueeze(0).repeat(2, 1, 1)  # (2, b, h_a)
        inputs = torch.cat((head, result_asr_features), dim=1)
        inputs = inputs.view(b, d, -1)
        (h_0, c_0) = audio_hidden, torch.zeros(*audio_hidden.shape).to(self.device)

        packed = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.rnn(packed)#, (h_0, c_0))
        seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        rnn_out = self.dropout(seq_unpacked)
        logits = self.classifier_rnn(rnn_out)  # (b,d,1)

        # Uncomment the following line for list-wise transformer setting
        # inputs = torch.cat((head, result_asr_features), dim=1)
        # inputs = inputs.view(b, d, -1)
        # trans_out = self.transformer_fuser(inputs, result_masks, result_audio)
        # trans_out = self.dropout(trans_out)
        # logits = self.classifier_trans(trans_out)  # (b,d,1)

        # Uncomment the following line for list-wise attention setting
        # audio_hidden = self.audio_encoder(result_audio)  # (b, h_a)
        # inputs = torch.cat((head, result_asr_features), dim=1)
        # inputs = inputs.view(b, d, -1)
        # decoder_hidden = (audio_hidden, torch.zeros(1, *audio_hidden.shape).to(self.device))
        #
        # rnn_out = torch.zeros(b, d, self.bert_dim, device=self.device)
        # for di in range(d):
        #     decoder_output, decoder_hidden = self.decoder(decoder_hidden, inputs[:, di, :], truth_embeddings, di)
        #     rnn_out[:, di, :] = decoder_output.squeeze(1)
        # rnn_out = self.dropout(rnn_out)
        # logits = self.classifier_rnn_att(rnn_out)  # (b,d,1)

        if self.loss_type == 'margin':
            return logits, truth_logits
        else:
            return logits

    def confidence_loss(self, logits, batch):
        """Compute confidence-based loss"""

        (result_ids, result_token_masks, result_masks, lengths, result_audio, result_wers, truth_ids,
         truth_token_masks, result_asr_features) = self.get_batch(batch)

        b, d, t = result_ids.shape

        total_loss = 0
        total_hyps = 0
        total_corrects = 0

        if self.loss_type == 'margin':
            logits, truth_logits = logits
            logits = logits.view(b, d, 1).squeeze(2) # (b, d)
            truth_logits = truth_logits.unsqueeze(1).repeat(1, d, 1).squeeze(2) # (b, d)

            for i in range(b):
                total_hyps += lengths[i]
                scores = logits[i][:lengths[i]]
                truth_scores = truth_logits[i][:lengths[i]]
                y = torch.ones(lengths[i]).to(self.device)
                total_loss += self.margin_loss(truth_scores, scores, y)

        else:

            logits = logits.view(b, d, 1).squeeze(2)
            for i in range(b):
                total_hyps += lengths[i]
                scores = logits[i][:lengths[i]]
                wers = result_wers[i][:lengths[i]]
                if self.loss_type == 'regression':
                    total_loss += self.mse(scores, wers)
                elif self.loss_type == 'bce':
                    confidence_scores = torch.where(wers == 0., torch.tensor(1.).to(self.device), torch.tensor(0.).to(self.device)).float()
                    total_loss += self.bce(scores, confidence_scores)
                    scores = self.sig(scores)
                    pred_scores = (scores >= 0.5).float()
                    total_corrects += torch.sum(torch.eq(pred_scores, confidence_scores).int())
                elif self.loss_type == 'bce_mwer':
                    wers_extend = wers.unsqueeze(0)
                    index = torch.argmin(wers_extend, dim=1)
                    confidence_scores = torch.zeros(1, lengths[i]).to(self.device).scatter_(1, index.unsqueeze(1), 1)
                    total_loss += self.bce(scores, confidence_scores.squeeze(0))
                    scores = self.sig(scores)
                    pred_scores = (scores >= 0.5).float()
                    total_corrects += torch.sum(torch.eq(pred_scores, confidence_scores.squeeze(0)).int())
                elif self.loss_type == 'ce':
                    wers_extend = wers.unsqueeze(0)
                    index = torch.argmin(wers_extend, dim=1)
                    scores = scores.unsqueeze(0)
                    total_loss += self.ce(scores, index.long())
                    scores_prob = nn.Softmax(dim=1)(scores)
                    preds = torch.argmax(scores_prob, dim=1)
                    total_corrects += (preds == index).int().squeeze(0)
                    total_hyps -= lengths[i]
                    total_hyps += 1
                elif self.loss_type == 'kl_div':
                    target = nn.Softmax(dim=1)(-wers.unsqueeze(0)) # (1, d)
                    scores = nn.Softmax(dim=1)(scores.unsqueeze(0)) # (1, d)
                    total_loss += self.kl(scores.log(), target)

        return total_loss, total_hyps, total_corrects


    def mwer_loss(self, logits, batch):
        """Compute minimum word error rate weighted cross entropy loss"""

        (result_ids, result_token_masks, result_masks, lengths, result_audio, result_wers, truth_ids,
         truth_token_masks, result_asr_features) = self.get_batch(batch)

        b, d, t = result_ids.shape
        logits = logits.view(b, d, 1).squeeze(2) # (b, d, 1)

        mwer_loss = 0
        for i in range(b):
            probs = logits[i][:lengths[i]]
            probs = nn.Sigmoid()(probs)
            probs = probs / probs.sum()
            wer_average = torch.mean(result_wers[i, :lengths[i]])
            per_sample_loss = torch.sum(probs * (result_wers[i, :lengths[i]] - wer_average))
            mwer_loss += per_sample_loss

        return mwer_loss

    def get_batch(self, batch):
        """Get batch to correct device"""

        new_batch = []
        for data in batch:
            new_batch.append(data.to(self.device))
        return tuple(new_batch)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(self.hidden_size * 2, 1)
        self.rnn_token = nn.LSTM(input_size=self.hidden_size*2+14,
                                 hidden_size=self.hidden_size,
                                 bidirectional=False,
                                 batch_first=True,
                                 num_layers=1)

    def forward(self, hidden, inputs, encoder_outputs, di):
        b, t, h = encoder_outputs.shape

        # repeat decoder hidden
        decoder_hidden = hidden[0].view(-1, self.hidden_size)  # (b,h)
        hidden_repeat = decoder_hidden.unsqueeze(1)  # (b,1,h)
        hidden_repeat = hidden_repeat.repeat(1, t, 1)  # (b,t,h)

        # attention
        attn_weights = self.attn(torch.cat((encoder_outputs, hidden_repeat), 2))  # (b,t,1)
        attn_weights = F.softmax(attn_weights, dim=1)  # (b,t,1)
        attn_applied = torch.bmm(encoder_outputs.transpose(2, 1), attn_weights).squeeze(2)  # (b,h)

        output = torch.cat((inputs, attn_applied), dim=1)  # (b,2h+14+h)

        # linear layer
        hidden = (decoder_hidden.unsqueeze(0), hidden[1])
        output = output.unsqueeze(1)  # (b,1,h)
        output, hidden = self.rnn_token(output, hidden)

        return output, hidden
