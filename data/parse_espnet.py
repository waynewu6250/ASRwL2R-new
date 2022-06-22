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
import json
import os
from collections import defaultdict
from re import L
import re
import pandas as pd
import numpy as np
import jiwer
from scipy.stats import rankdata
import pickle
import argparse

N = 30

def get_wer_rank(wers):
    scores = [1-float(wer) for wer in wers]
    return rankdata(scores, method='max')

def parse_raw_data(folder):
    """Parse the raw data into a dictionary containing all information.
    score_family format:
    {'score': {'utt_id1': [hyp1_score, hyp2_score, ...]},
     'decoder': {'utt_id1': [hyp1_score, hyp2_score, ...]},
     ...}
    """

    def store_sentence(score_family, path, loc):
        with open(path, 'r') as f:
            for line in f:
                lines = line.split('\t')
                utt_id = lines[1].strip()[1:-1]
                utt_id = '-'.join(utt_id.split('-')[2:])
                score_family[loc][utt_id].append(lines[0].lower())

    score_family = {key: defaultdict(list) for key in ('score', 'decoder', 'ctc', 'lm', 'hyp', 'truth')}

    # references
    path = os.path.join(folder,'references/ref_wer.trn')
    store_sentence(score_family, path, 'truth')
    
    # tokenized_hypotheses
    for i in range(1, N+1):
        path = os.path.join(folder,'tokenized_hypotheses/{}best_recog/hyp_wer.trn'.format(str(i)))
        store_sentence(score_family, path, 'hyp')
    
    # score
    for i in range(1, N+1):
        subfolder = os.path.join(folder,'raw_hypotheses/{}best_recog/'.format(str(i)))

        # score
        path = os.path.join(subfolder, 'score')
        with open(path, 'r') as f:
            for line in f:
                utt_id = line.split(' ')[0]
                score = float(re.search('tensor\((.*)\)', line.split(' ')[1]).group(1))
                score_family['score'][utt_id].append(score)
        
        # other scores
        path = os.path.join(subfolder, 'scores')
        with open(path, 'r') as f:
            for line in f:
                utt_id = line.split(' ')[0]
                decoder = float(re.search('tensor\((.*)\)', line.split(' ')[2]).group(1))

                score_family['decoder'][utt_id].append(decoder)
                ctc = float(re.search('tensor\((.*)\)', line.split(' ')[4]).group(1))

                score_family['ctc'][utt_id].append(ctc)
                lm = float(re.search('tensor\((.*)\)', line.split(' ')[6]).group(1))

                score_family['lm'][utt_id].append(lm)
    
    return score_family

def parse_data(score_family, collections, group_id, bce_dic=None, bce_mwer_dic=None, ce_dic=None):

    for utt_id in score_family['hyp'].keys():
        wers = []
        counter = 0
        for j in range(N):
            collections['utt_id'].append(utt_id)
            collections['hyp_id'].append(utt_id+'-'+str(j+1))
            for key in ('score', 'decoder', 'ctc', 'lm', 'hyp'):
                collections[key].append(score_family[key][utt_id][j])
            collections['truth'].append(score_family['truth'][utt_id][0])
            hyp = score_family['hyp'][utt_id][j]
            truth = score_family['truth'][utt_id][0]
            if truth == '':
                mer = jiwer.compute_measures(hyp, hyp)
            else:
                mer = jiwer.compute_measures(truth, hyp)
            collections['#subErr'].append(mer['substitutions'])
            collections['#insErr'].append(mer['insertions'])
            collections['#delErr'].append(mer['deletions'])
            collections['WER'].append(mer['wer'])
            wers.append(mer['wer'])
            if bce_dic:
                collections['bce_score'].append(bce_dic[utt_id][counter])
                collections['bce_mwer_score'].append(bce_mwer_dic[utt_id][counter])
                # collections['ce_score'].append(ce_dic[utt_id][counter])
            counter += 1
        
        ranks = get_wer_rank(wers)
        for i in range(len(wers)):
            collections['rank'].append(ranks[i])
            collections['group_id'].append(group_id)
        
        group_id += 1
    
    return collections, group_id


#########################################################################################################

if not os.path.exists('espnet_parsed/'):
    os.mkdir('espnet_parsed/')

root = './espnet/simpleoier_librispeech_asr_train_asr_conformer7_hubert_ll60k_large_raw_en_bpe5000_sp/decode_asr_lm_lm_train_lm_transformer2_en_bpe5000_valid.loss.ave_asr_model_valid.acc.ave/'

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='phase_2', dest='type', help='phase_1 | phase_2')
args = parser.parse_args()

#####################
# Parse the training data

train_bce_dic, test_bce_dic = None, None
train_bce_mwer_dic, test_bce_mwer_dic = None, None
train_ce_dic, test_ce_dic = None, None

if args.type == 'phase_2':
    print('Load bert confidence model scores...')
    with open('espnet_parsed/train-all_bert_bert_rescorer_bce_listwise.pkl', 'rb') as f:
        train_bce_dic = pickle.load(f)
    with open('espnet_parsed/train-all_bert_bert_rescorer_bce_mwer_listwise.pkl', 'rb') as f:
        train_bce_mwer_dic = pickle.load(f)
    
collections = defaultdict(list)
group_id = 0
for folder in os.listdir(root):
    if folder in ['dev_clean', 'dev_other']:
        print('parse: ', folder)
        score_family = parse_raw_data(os.path.join(root, folder))
        collections, group_id = parse_data(score_family, collections, group_id, train_bce_dic, train_bce_mwer_dic, train_ce_dic)
data = pd.DataFrame(collections)
data['hyp_length'] = data['hyp'].map(lambda x: len(x.split(' ')))
data['#refWord'] = data['truth'].map(lambda x: len(x.split(' ')))
print('Total utterances: ', data.loc[len(data)-1]['group_id']+1)
print('Total hypotheses: ', len(data))
# print(data.loc[:100])
data.to_csv('espnet_parsed/train-all.csv')

# #####################
# Parse the testing data

if args.type == 'phase_2':
    with open('espnet_parsed/test-clean_bert_bert_rescorer_bce_listwise.pkl', 'rb') as f:
        test_bce_dic = pickle.load(f)
    with open('espnet_parsed/test-clean_bert_bert_rescorer_bce_mwer_listwise.pkl', 'rb') as f:
        test_bce_mwer_dic = pickle.load(f)

folders = ['test_clean', 'test_other']
for folder in folders:
    group_id = 0
    collections = defaultdict(list)
    print('parse: ', folder)
    score_family = parse_raw_data(os.path.join(root, folder))
    collections, group_id = parse_data(score_family, collections, group_id, test_bce_dic, test_bce_mwer_dic, test_ce_dic)
    data = pd.DataFrame(collections)
    data['hyp_length'] = data['hyp'].map(lambda x: len(x.split(' ')))
    data['#refWord'] = data['truth'].map(lambda x: len(x.split(' ')))
    print('Total utterances: ', data.loc[len(data)-1]['group_id']+1)
    print('Total hypotheses: ', len(data))
    # print(data.iloc[:300][['hyp','score','bert_score','bce_score','bce_mwer_score']])
    data.to_csv('espnet_parsed/{}.csv'.format(folder))
