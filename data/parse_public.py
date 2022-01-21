import json
import os
from collections import defaultdict
import pandas as pd
import numpy as np
import jiwer
from scipy.stats import rankdata
import pickle
import argparse

if not os.path.exists('libri_subset/'):
    os.mkdir('libri_subset/')

N = 30

def get_wer_rank(wers):
    scores = [1-float(wer) for wer in wers]
    return rankdata(scores, method='max')

def parse_data_train(file, collections, group_id, score_dic=None, bce_dic=None, bce_mwer_dic=None, ce_dic=None):

    with open(file, 'r') as f:
        dic = json.load(f)

    # Get the smallest hypothesis id for each data
    # index = []
    # for k, v in dic.items():
    #     truth = v['ref']
    #     wers = []
    #     for hyp_id, hyps in v.items():
    #         if hyp_id == 'ref':
    #             continue
    #         mer = jiwer.compute_measures(truth, hyps['text'])
    #         wers.append(mer['wer'])
    #     index.append(np.argmin(wers))
    
    for k, v in dic.items():
        # k: data utterance id
        truth = v['ref']
        v.pop('ref')
        wers = []
        counter = 0
        counter1 = 0
        counter2 = 0
        counter3 = 0
        for iter, (hyp_id, hyps) in enumerate(v.items()):
            idd = int(hyp_id.split('_')[1])
            # if index[iter] < N:
            #     criteria = idd < N+1
            # else:
            #     criteria = idd < N or idd == index[iter]

            # hyp_id: hypothesis id
            if idd < N+1:
                collections['data_id'].append(k+'_1')
                kk = k+'_1'
                if bce_dic:
                    collections['bce_score'].append(bce_dic[kk][counter1])
                    collections['bce_mwer_score'].append(bce_mwer_dic[kk][counter1])
                counter1 += 1
            # elif int(hyp_id.split('_')[1]) < 61:
            #     collections['data_id'].append(k+'_2')
            #     kk = k+'_2'
            #     if bce_dic:
            #         collections['bce_score'].append(bce_dic[kk][counter2])
            #         collections['bce_mwer_score'].append(bce_mwer_dic[kk][counter2])
            #     counter2 += 1
            # elif int(hyp_id.split('_')[1]) < 91:
            #     collections['data_id'].append(k+'_3')
            #     kk = k+'_3'
            #     if bce_dic:
            #         collections['bce_score'].append(bce_dic[kk][counter3])
            #         collections['bce_mwer_score'].append(bce_mwer_dic[kk][counter3])
            #     counter3 += 1
            else:
                continue
            collections['hyp_id'].append(hyp_id)
            collections['score'].append(hyps['score'])
            collections['hyp'].append(hyps['text'])
            collections['truth'].append(truth)
            if truth == '':
                mer = jiwer.compute_measures(hyps['text'], hyps['text'])
            else:
                mer = jiwer.compute_measures(truth, hyps['text'])
            collections['#subErr'].append(mer['substitutions'])
            collections['#insErr'].append(mer['insertions'])
            collections['#delErr'].append(mer['deletions'])
            collections['WER'].append(mer['wer'])
            wers.append(mer['wer'])
            counter += 1
            
        for i in range(1):
            ranks = get_wer_rank(wers[i*N:(i+1)*N])
            collections['rank'].extend(ranks)
            collections['group_id'].extend([group_id]*N)
            group_id += 1
        # ranks = get_wer_rank(wers)
        # for i in range(len(wers)):
        #     collections['rank'].append(ranks[i])
        #     collections['group_id'].append(group_id)
        
        group_id += 1
    
    return collections, group_id

def parse_data_test(file, collections, group_id, score_dic=None, bce_dic=None, bce_mwer_dic=None, ce_dic=None):

    with open(file, 'r') as f:
        dic = json.load(f)
    
    # Get the smallest hypothesis id for each data
    # index = []
    # for k, v in dic.items():
    #     truth = v['ref']
    #     wers = []
    #     for hyp_id, hyps in v.items():
    #         if hyp_id == 'ref':
    #             continue
    #         mer = jiwer.compute_measures(truth, hyps['text'])
    #         wers.append(mer['wer'])
    #     index.append(np.argmin(wers))
    
    for k, v in dic.items():
        # k: data utterance id
        truth = v['ref']
        v.pop('ref')
        wers = []
        counter = 0

        for iter, (hyp_id, hyps) in enumerate(v.items()):
            # hyp_id: hypothesis id
            idd = int(hyp_id.split('_')[1])
            # if index[iter] < N:
            #     criteria = idd < N+1
            # else:
            #     criteria = idd < N or idd == index[iter]

            if idd < N+1: # 31
                collections['data_id'].append(k)
                collections['hyp_id'].append(hyp_id)
                collections['score'].append(hyps['score'])
                collections['hyp'].append(hyps['text'])
                collections['truth'].append(truth)
                if truth == '':
                    mer = jiwer.compute_measures(hyps['text'], hyps['text'])
                else:
                    mer = jiwer.compute_measures(truth, hyps['text'])
                collections['#subErr'].append(mer['substitutions'])
                collections['#insErr'].append(mer['insertions'])
                collections['#delErr'].append(mer['deletions'])
                collections['WER'].append(mer['wer'])
                wers.append(mer['wer'])
                if bce_dic:
                    collections['bert_score'].append(-score_dic[k][counter])
                    collections['bce_score'].append(bce_dic[k][counter])
                    collections['bce_mwer_score'].append(bce_mwer_dic[k][counter])
                    collections['ce_score'].append(ce_dic[k][counter])
                counter += 1

        ranks = get_wer_rank(wers)
        for i in range(len(wers)):
            collections['rank'].append(ranks[i])
            collections['group_id'].append(group_id)
        
        group_id += 1
    
    return collections, group_id


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='phase_2', dest='type', help='phase_1 | phase_2')
args = parser.parse_args()

#####################
# Parse the training data

train_score_dic, test_score_dic = None, None
train_bce_dic, test_bce_dic = None, None
train_bce_mwer_dic, test_bce_mwer_dic = None, None
train_ce_dic, test_ce_dic = None, None

if args.type == 'phase_2':
    with open('libri_subset/train-all_bert_score_pretrain.pkl', 'rb') as f:
        train_score_dic = pickle.load(f)
    with open('libri_subset/train-all_bert_bert_rescorer_bce_listwise.pkl', 'rb') as f:
        train_bce_dic = pickle.load(f)
    with open('libri_subset/train-all_bert_bert_rescorer_bce_mwer_listwise.pkl', 'rb') as f:
        train_bce_mwer_dic = pickle.load(f)
    with open('libri_subset/train-all_bert_bert_rescorer_ce_listwise.pkl', 'rb') as f:
        train_ce_dic = pickle.load(f)
    
collections = defaultdict(list)
group_id = 0
for file in os.listdir('./'):
    if '.json' in file and 'test-clean' not in file:
        print('parse: ', file)
        collections, group_id = parse_data_train(file, collections, group_id, train_score_dic, train_bce_dic, train_bce_mwer_dic, train_ce_dic)
data = pd.DataFrame(collections)
data['hyp_length'] = data['hyp'].map(lambda x: len(x.split(' ')))
data['#refWord'] = data['truth'].map(lambda x: len(x.split(' ')))
print(len(data))
data.to_csv('libri_subset/train-all-test.csv')

#####################
# Parse the testing data

if args.type == 'phase_2':
    with open('libri_subset/test-clean_bert_score_pretrain.pkl', 'rb') as f:
        test_score_dic = pickle.load(f)
    with open('libri_subset/test-clean_bert_bert_rescorer_bce_listwise.pkl', 'rb') as f:
        test_bce_dic = pickle.load(f)
    with open('libri_subset/test-clean_bert_bert_rescorer_bce_mwer_listwise.pkl', 'rb') as f:
        test_bce_mwer_dic = pickle.load(f)
    with open('libri_subset/test-clean_bert_bert_rescorer_ce_listwise.pkl', 'rb') as f:
        test_ce_dic = pickle.load(f)

collections = defaultdict(list)
group_id = 0
collections, group_id = parse_data_test('test-clean.am.json', collections, group_id, test_score_dic, test_bce_dic, test_bce_mwer_dic, test_ce_dic)
data = pd.DataFrame(collections)
data['hyp_length'] = data['hyp'].map(lambda x: len(x.split(' ')))
data['#refWord'] = data['truth'].map(lambda x: len(x.split(' ')))
print(len(data))
# print(data.iloc[:300][['hyp','score','bert_score','bce_score','bce_mwer_score']])
data.to_csv('libri_subset/test-clean-test.csv')