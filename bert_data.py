"""
Copyright 2022 Amazon.com, Inc. and its affiliates. All Rights Reserved.
 
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
import torch as t
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from transformers import BertTokenizer
import re
import numpy as np
from utils import get_groups
from config import opt
from data.unstructured_dataset import UnStructuredDataset
from time import time
import pickle

class SymmetricLog1pScaler(object):
    """Symmetric Log1p Transformation"""

    @staticmethod
    def fit_transform(X):
        return np.sign(X) * np.log(1.0 + np.abs(X))

class CoreDataset(Dataset):

    def __init__(self, dataset, utterance_mode='group', embed='bert', audio_dic=None):
        """Torch dataset for bert data loader

        :parameter:
            utterances: a list of utterance tuples  [(nbest_ids_list1, nbest_masks_list1),
                                                     (nbest_ids_list2, nbest_masks_list2), ...]
            audio_dic: a dictionary of {utt_id: audio embeddings}
        """
        self.maxlen = 20
        self.REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
        self.BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
        if embed == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.data = dataset.data
        self.utterance_mode = utterance_mode
        self.sparse_features = dataset.data_for_use
        # for column in ["neg_confidence", "neg_confidence_stable", "neg_directedness", "neg_directedness_stable"]:
        #     self.sparse_features.loc[:][column] = self.sparse_features.loc[:][column].replace(np.NaN,
        #                                                                                       self.sparse_features[
        #                                                                                           column].mean())

        # Set a new column by replacing truth to rewrite if truth is None
        # self.data['truth_update'] = self.data['rewrite'] #np.where(self.data['truth'] != '', self.data['truth'], self.data['rewrite'])

        if self.utterance_mode == 'single':
            # Group-agnostic information
            self.input_ids, self.attention_masks = self.tokenize_and_pad_data(dataset.raw_texts)

        elif self.utterance_mode == 'group':
            # Group-aware information:
            query_groups = get_groups(self.data.utt_id)
            self.utterances = []
            self.truths = []
            self.audio_embeddings = []
            self.wers = []
            self.asr_features = []
            for i, (qid, a, b) in enumerate(query_groups):

                # little mismatch in large dataset
                # if qid not in dataset.audio_dic:
                #     continue

                # hypothesis
                hypotheses = self.data.loc[a:b-1]['hyp'].values
                wers = [float(wer) for wer in self.data.loc[a:b-1]['WER'].values] # make sure wer is float number
                tokenized_data = self.tokenize_and_pad_data(hypotheses)
                self.utterances.append(tokenized_data)

                # ground truth
                tokenized_truth = self.tokenize_and_pad_data([self.data.loc[a]['truth']])
                self.truths.append((tokenized_truth[0][0], tokenized_truth[1][0]))
                # self.audio_embeddings.append(dataset.audio_dic[qid].mean(axis=0)) # average them together to make it (b,h)
                self.wers.append(wers)

                # asr feature
                hyp_asr_feature = self.sparse_features.loc[a:b - 1].values  # (b, num_features)
                hyp_asr_feature = SymmetricLog1pScaler().fit_transform(hyp_asr_feature)
                self.asr_features.append(hyp_asr_feature)
            self.audio_embeddings = t.zeros(len(self.utterances), 768)

    def __getitem__(self, index):
        """Get item"""

        if self.utterance_mode == 'single':
            # caps
            caps = self.input_ids[index]
            caps = t.tensor(caps)

            # masks
            masks = t.tensor(self.attention_masks[index])

            return caps, masks

        elif self.utterance_mode == 'group':
            return self.utterances[index], self.audio_embeddings[index], self.wers[index], self.truths[index], self.asr_features[index]

    def __len__(self):
        """Get length"""
        return len(self.input_ids) if self.utterance_mode == 'single' else len(self.utterances)


    def text_prepare(self, text):
        """ prepare text data
        :param text: a string
        :return: modified string
        """

        text = text.lower()  # lowercase text
        text = re.sub(self.REPLACE_BY_SPACE_RE, ' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = re.sub(self.BAD_SYMBOLS_RE, '', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
        text = re.sub(r"[ ]+", " ", text)
        text = re.sub(r"\!+", "!", text)
        text = re.sub(r"\,+", ",", text)
        text = re.sub(r"\?+", "?", text)

        text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(text)
        tokenized_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        text = tokenized_ids

        return text

    def tokenize_and_pad_data(self, hypotheses):
        """ prepare text data
        :param hypotheses: a list of hypothesis sentences
        :return: tuple of tokenized ids and attention masks
        """
        tokenized_data = [self.text_prepare(str(text)) for text in hypotheses]
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(tokenized_data, maxlen=self.maxlen, dtype="long",
                                                                  truncating="post", padding="post")
        attention_masks = []
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)
        return (input_ids, attention_masks)



def collate_fn(batch):
    """collate function for batching the inputs"""

    utterances, audio_embeddings, wers, truths, asr_features = zip(*batch)

    lengths = [token_ids.shape[0] for token_ids, attention_masks in utterances] # number of hypotheses for each utterance
    lengths = t.LongTensor(lengths)

    max_len = max([token_ids.shape[0] for token_ids, attention_masks in utterances])
    max_dim = max([token_ids.shape[1] for token_ids, attention_masks in utterances])
    result_ids = t.zeros((len(utterances), max_len, max_dim)).long()
    result_token_masks = t.zeros((len(utterances), max_len, max_dim)).long()
    result_masks = t.zeros((len(utterances), max_len)).long()
    result_audio = t.zeros((len(utterances), 768))
    result_wers = t.zeros((len(utterances), max_len))
    result_asr_features = t.zeros((len(utterances)), max_len, asr_features[0].shape[1])

    for i in range(len(utterances)):
        len1 = utterances[i][0].shape[0]
        dim1 = utterances[i][0].shape[1]
        result_ids[i, :len1, :dim1] = t.Tensor(utterances[i][0])
        result_token_masks[i, :len1, :dim1] = t.Tensor(utterances[i][1])
        result_asr_features[i, :len1, :] = t.Tensor(asr_features[i])
        for j in range(lengths[i]):
            result_masks[i][j] = 1
            result_wers[i][j] = wers[i][j]
        result_audio[i, :] = t.tensor(audio_embeddings[i])

    truth_ids, truth_token_masks = zip(*truths)
    truth_ids = t.LongTensor(truth_ids)
    truth_token_masks = t.LongTensor(truth_token_masks)


    return result_ids, result_token_masks, result_masks, lengths, result_audio, result_wers, truth_ids, truth_token_masks, result_asr_features


def get_dataloader(dataset, utterance_mode='group', pretrain_embed='bert'):
    """Get the torch dataloader given an unstructured dataset"""
    dataset = CoreDataset(dataset, utterance_mode, pretrain_embed)
    batch_size = opt.batch_size

    if utterance_mode == 'single':
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False)
    elif utterance_mode == 'group':
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=lambda x: collate_fn(x))
    return dataloader

if __name__ == '__main__':
    # sanity check
    start = time()
    dataset = UnStructuredDataset(opt.test_path, opt=opt)
    loader = get_dataloader(dataset, utterance_mode='group')

    for i, (result_ids, result_token_masks, result_masks,
            lengths, result_audio, result_wers,
            truth_ids, truth_token_masks, result_asr_features) in enumerate(loader):
        j = 0
        print(result_ids[j])
        print(result_token_masks[j])
        print(result_masks[j])
        print(lengths[j])
        print(result_audio[j, :15])
        print(result_wers[j])
        print(truth_ids[j])
        print(truth_token_masks[j])
        print(result_asr_features[j])
        print('----------')
    print('time :', time()-start)
