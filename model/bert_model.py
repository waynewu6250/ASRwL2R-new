import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, AutoModel, BertConfig, AdamW, BertForPreTraining, BertForMaskedLM, AutoConfig, AutoModelForMaskedLM


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertEmbedding(nn.Module):

    def __init__(self, checkpoint_path):
        """Construct a BERT embedding model.
        """
        super(BertEmbedding, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Another way to import pretrained model
        # config = BertConfig.from_json_file('./uncased_L-12_H-768_A-12/bert_config.json')
        # config.output_hidden_states = True
        # self.bert_embedding = BertForPreTraining.from_pretrained('./uncased_L-12_H-768_A-12/bert_model.ckpt.index', from_tf=True, config=config)

        # model
        config = AutoConfig.from_pretrained('bert-base-uncased')
        config.output_hidden_states = True
        self.bert_embedding = AutoModelForMaskedLM.from_pretrained('bert-base-uncased', config=config)

        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, labels=None, seg_tensors=None):
        """
        Main forward function of BertEmbedding Model
        :param input_ids: (b, t) tokenized ids of input text data
        :param attention_mask: (b, t) binary mask for bert input
        :return outputs:
            last_hidden_states: (b, t, h)
            pooled_output: (b, h), from output of a linear classifier + tanh
            hidden_states: 13 x (b, t, h), embed to last layer embedding
            attentions: 12 x (b, num_heads, t, t)
        """
        outputs = self.bert_embedding(input_ids, token_type_ids=seg_tensors, attention_mask=attention_mask, labels=labels)
        return outputs