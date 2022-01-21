#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
Given the neural ranker, compute nDCG values.
"""

import torch

from model.ptranking.data.data_utils import LABEL_TYPE
from model.ptranking.metric.adhoc_metric import torch_nDCG_at_k, torch_nDCG_at_ks
import numpy as np
import pandas as pd
from jiwer import wer

def ndcg_at_k(ranker=None, test_data=None, k=10, label_type=LABEL_TYPE.MultiLabel, gpu=False, device=None):
    '''
    There is no check based on the assumption (say light_filtering() is called) that each test instance Q includes at least k documents,
    and at least one relevant document. Or there will be errors.
    '''
    sum_ndcg_at_k = torch.zeros(1)
    cnt = torch.zeros(1)
    already_sorted = True if test_data.presort else False
    for qid, batch_ranking, batch_labels in test_data: # _, [batch, ranking_size, num_features], [batch, ranking_size]
        if batch_labels.size(1) < k: continue # skip the query if the number of associated documents is smaller than k

        if gpu: batch_ranking = batch_ranking.to(device)
        batch_rele_preds = ranker.predict(batch_ranking)
        if gpu: batch_rele_preds = batch_rele_preds.cpu()

        _, batch_sorted_inds = torch.sort(batch_rele_preds, dim=1, descending=True)

        batch_sys_sorted_labels = torch.gather(batch_labels, dim=1, index=batch_sorted_inds)
        if already_sorted:
            batch_ideal_sorted_labels = batch_labels
        else:
            batch_ideal_sorted_labels, _ = torch.sort(batch_labels, dim=1, descending=True)

        batch_ndcg_at_k = torch_nDCG_at_k(batch_sys_sorted_labels=batch_sys_sorted_labels,
                                          batch_ideal_sorted_labels=batch_ideal_sorted_labels,
                                          k = k, label_type=label_type)

        sum_ndcg_at_k += torch.squeeze(batch_ndcg_at_k) # default batch_size=1 due to testing data
        cnt += 1

    avg_ndcg_at_k = sum_ndcg_at_k/cnt
    return  avg_ndcg_at_k

def ndcg_at_ks(ranker=None, test_data=None, ks=[1, 5, 10], label_type=LABEL_TYPE.MultiLabel, gpu=False, device=None):
    '''
    There is no check based on the assumption (say light_filtering() is called)
    that each test instance Q includes at least k(k=max(ks)) documents, and at least one relevant document.
    Or there will be errors.
    '''
    sum_ndcg_at_ks = torch.zeros(len(ks))
    cnt = torch.zeros(1)
    already_sorted = True if test_data.presort else False

    for qid, batch_ranking, batch_labels in test_data: # _, [batch, ranking_size, num_features], [batch, ranking_size]

        if gpu: batch_ranking = batch_ranking.to(device)
        batch_rele_preds = ranker.predict(batch_ranking)
        if gpu: batch_rele_preds = batch_rele_preds.cpu()

        _, batch_sorted_inds = torch.sort(batch_rele_preds, dim=1, descending=True)

        batch_sys_sorted_labels = torch.gather(batch_labels, dim=1, index=batch_sorted_inds)
        if already_sorted:
            batch_ideal_sorted_labels = batch_labels
        else:
            batch_ideal_sorted_labels, _ = torch.sort(batch_labels, dim=1, descending=True)

        batch_ndcg_at_ks = torch_nDCG_at_ks(batch_sys_sorted_labels=batch_sys_sorted_labels,
                                            batch_ideal_sorted_labels=batch_ideal_sorted_labels,
                                            ks=ks, label_type=label_type)

        # default batch_size=1 due to testing data
        sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, torch.squeeze(batch_ndcg_at_ks, dim=0))
        cnt += 1

    avg_ndcg_at_ks = sum_ndcg_at_ks/cnt
    return avg_ndcg_at_ks

def calc_wer(ranker=None, test_data=None, gpu=False, device=None, test_data_df=None):

    preds = []
    for qid, batch_ranking, batch_labels in test_data: # _, [batch, ranking_size, num_features], [batch, ranking_size]

        if gpu: batch_ranking = batch_ranking.to(device)
        batch_rele_preds = ranker.predict(batch_ranking)
        if gpu: batch_rele_preds = batch_rele_preds.cpu()
        pred = batch_rele_preds.squeeze(0).detach().numpy().tolist()

        preds += pred

    baseline_data = test_data_df[["acoustic_model", "sub_decoder_0_G-pryon-BG_SLM.compact.fst",
                    "sub_decoder_0_sub_lm_bg_slm.hash_quant.lm", "nlm"]]
    weight = pd.DataFrame(pd.Series([0.4, 0.25, 0.25, 0.75], index=baseline_data.columns, name=0))
    test_data_df['score'] = -baseline_data.dot(weight)
    rank_score_weight = test_data_df.score.to_numpy()
    alpha = [0, 0.0001, 0.0004, 0.0005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0]
    best_wer_score = 1
    for a in alpha:
        rank_score = rank_score_weight * a + np.array(preds)  # + 0.01 * self.test_data['mulan_distillbert_mlm_score'].values  - self.test_data.trans_score_mulan.to_numpy().astype(np.float32)
        wer_score = calculate_best_wer(test_data_df.utt_id,
                                       rank_score,
                                       test_data_df.truth.to_numpy(),
                                       test_data_df.hyp.to_numpy(),
                                       test_data_df)
        if wer_score < best_wer_score:
            best_wer_score = wer_score
            best_alpha = a
    print('Best WER score from wer file with {}: {:.5f}'.format(best_alpha, best_wer_score))

    # best_wer_scores = calculate_best_wer(test_data_df.utt_id,
    #                                      preds,
    #                                      test_data_df.truth.to_numpy(),
    #                                      test_data_df.hyp.to_numpy(),
    #                                      test_data_df)
    # print('WER: {}'.format(best_wer_scores))

def calculate_best_wer(qids, preds, reference, hypothesis, test_data):
    """Predicts the scores for the test dataset and calculates the best WER value based on ranking."""

    query_groups = get_groups(qids)  # (qid,from,to)
    references = []
    hypotheses = []
    references_without_wake_word = []
    hypotheses_without_wake_word = []
    hypotheses_id = []
    for qid, a, b in query_groups:
        predicted_sorted_indexes = np.argsort(preds[a:b])[::-1]  # from highest to lowest
        best_id = predicted_sorted_indexes[0]
        real_id = list(range(a, b))[best_id]

        ref = str(reference[real_id])
        hyp = str(hypothesis[real_id])
        references.append(ref)
        hypotheses.append(hyp)

        if remove_wake_words(ref) or remove_wake_words(hyp):
            references_without_wake_word.append(remove_wake_words(ref))
            hypotheses_without_wake_word.append(remove_wake_words(hyp))

        hypotheses_id.append(real_id)

    print('NWWER score: %.5f' % wer(references_without_wake_word, hypotheses_without_wake_word))

    # Calculate from wer file
    nbest_selected = test_data.loc[hypotheses_id]
    nbest_selected['total_error'] = nbest_selected['#subErr'].astype('float64') + nbest_selected['#insErr'].astype(
        'float64') + nbest_selected['#delErr'].astype('float64')
    wer_score = sum(nbest_selected['total_error']) / sum(nbest_selected['#refWord'].astype('float64'))
    print('WER score from wer file: %.5f' % wer_score)
    return wer_score

def get_groups(qids):
    """Make an iterator of query groups on the provided list of query ids."""

    prev_qid = None
    prev_limit = 0
    total = 0

    for i, qid in enumerate(qids):
        total += 1
        if qid != prev_qid:
            if i != prev_limit:
                yield (prev_qid, prev_limit, i)
            prev_qid = qid
            prev_limit = i

    if prev_limit != total:
        yield (prev_qid, prev_limit, total)

def remove_wake_words(string):
    """ Remove wake words for a given string
    :param string: a string
    :return string: a string without wake words
    """
    wake_words = ['alexa', 'amazon', 'echo', 'computer']
    querywords = string.split()
    resultwords = [word for word in querywords if word.lower() not in wake_words]
    result = ' '.join(resultwords)

    return result





