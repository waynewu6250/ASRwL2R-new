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
import numpy as np
import collections
import lightgbm as lgb
from jiwer import wer
import json
import re

def calculate_1best_wer(qids, reference, hypothesis, test_data):
    """Predicts the scores for the test dataset and calculates the 1-best WER value.
    :param qids:numpy array of utterance id
    :param reference: reference text
    :param hypothesis: hypothesis text
    :return: WER score
    """

    query_groups = get_groups(qids)  # (qid,from,to)
    references = []
    hypotheses = []
    references_without_wake_word = []
    hypotheses_without_wake_word = []
    hypotheses_id = []
    for qid, a, b in query_groups:
        real_id = a
        ref = str(reference[real_id])
        hyp = str(hypothesis[real_id])
        references.append(ref)
        hypotheses.append(hyp)

        if remove_wake_words(ref) or remove_wake_words(hyp):
            references_without_wake_word.append(remove_wake_words(ref))
            hypotheses_without_wake_word.append(remove_wake_words(hyp))

        hypotheses_id.append(real_id)

    # Compare with raw truth (depreciated)
    # print('WER score from raw truth: %.5f' % wer(references, hypotheses))
    print('NWWER score: %.5f' % wer(references_without_wake_word, hypotheses_without_wake_word))

    # Calculate from wer file
    nbest_selected = test_data.loc[hypotheses_id]
    nbest_selected['total_error'] = nbest_selected['#subErr'].astype('float64')+nbest_selected['#insErr'].astype('float64')+nbest_selected['#delErr'].astype('float64')
    wer_score = sum(nbest_selected['total_error']) / sum(nbest_selected['#refWord'].astype('float64'))
    print('WER score from wer file: %.5f' % wer_score)

def calculate_min_wer(qids, preds, test_data):
    """Calculate the wer based on predicted ranks.
    :param qids:numpy array of utterance id
    :param preds: predicted rank
    :param test_data: test dataframe
    :return: WER score
    """

    query_groups = get_groups(qids)  # (qid,from,to)
    hypotheses_id = []
    for qid, a, b in query_groups:
        real_id = get_best_id(preds, a, b)
        hypotheses_id.append(real_id)
    nbest_selected = test_data.loc[hypotheses_id]
    nbest_selected['total_error'] = nbest_selected['#subErr'].astype('float64') + nbest_selected['#insErr'].astype(
        'float64') + nbest_selected['#delErr'].astype('float64')
    return sum(nbest_selected['total_error']) / sum(nbest_selected['#refWord'].astype('float64'))


def calculate_best_wer(qids, preds, reference, hypothesis, test_data, run_error=False):
    """Predicts the scores for the test dataset and calculates the best WER value based on ranking.
    :param qids:numpy array of utterance id
    :param preds: predicted rank
    :param reference: reference text
    :param hypothesis: hypothesis text
    :param test_data: test dataframe
    :param run_error: whether to produce error analysis file
    :return: WER score
    """

    oracle_scores = -test_data.WER.to_numpy().astype(np.float32)
    query_groups = get_groups(qids)  # (qid,from,to)
    references = []
    hypotheses = []
    all_hypotheses = []
    all_wers = []
    references_without_wake_word = []
    hypotheses_without_wake_word = []
    hypotheses_id = []
    # weights_id = []
    best1_id = []
    oracle_id = []
    rewrite_id = []
    for qid, a, b in query_groups:

        # score
        real_id = get_best_id(preds, a, b)

        ref = str(reference[real_id])
        hyp = str(hypothesis[real_id])
        references.append(ref)
        hypotheses.append(hyp)
        all_hypotheses.append([str(h) for h in hypothesis[a:b]])
        all_wers.append([float(w) for w in test_data.loc[a:b-1]['WER']])

        if remove_wake_words(ref) or remove_wake_words(hyp):
            references_without_wake_word.append(remove_wake_words(ref))
            hypotheses_without_wake_word.append(remove_wake_words(hyp))

        hypotheses_id.append(real_id)

        # oracle
        real_id = get_best_id(oracle_scores, a, b)
        oracle_id.append(real_id)

        # weight score
        # weight_id = get_best_id(preds2, a, b)
        # weights_id.append(weight_id)

        # 1best
        best1_id.append(a)
        rewrite_id.append(b-1)

    # Compare with raw truth (depreciated)
    # print('WER score from raw truth: %.5f' % wer(references, hypotheses))
    print('NWWER score: %.5f' % wer(references_without_wake_word, hypotheses_without_wake_word))

    # Calculate from wer file
    nbest_selected = test_data.loc[hypotheses_id]
    nbest_selected['total_error'] = nbest_selected['#subErr'].astype('float64')+nbest_selected['#insErr'].astype('float64')+nbest_selected['#delErr'].astype('float64')
    wer_score = sum(nbest_selected['total_error']) / sum(nbest_selected['#refWord'].astype('float64'))
    print('WER score from wer file: %.5f' % wer_score)

    # Produce error analysis texts for comparison
    if run_error:
        print('-------------------')
        # Compare 1-best and current model hypothesis: model better case
        generate_error_txt(test_data, hypotheses_id, best1_id, rewrite_id, reference, hypothesis,
                           '/home/ec2-user/model/error_analysis_better_than_1best.txt', '1best', all_hypotheses, all_wers, is_better=True
                           )

        # Compare 1-best and current model hypothesis: model worse case
        generate_error_txt(test_data, hypotheses_id, best1_id, rewrite_id, reference, hypothesis,
                           '/home/ec2-user/model/error_analysis_worse_than_1best.txt', '1best', all_hypotheses, all_wers, is_better=False)

        # Compare WER and current model hypothesis: oracle case
        generate_error_txt(test_data, hypotheses_id, oracle_id, rewrite_id, reference, hypothesis,
                           '/home/ec2-user/model/error_analysis_oracle.txt', 'oracle', all_hypotheses, all_wers, is_better=False)

        # generate_error_txt(test_data, hypotheses_id, weights_id, reference, hypothesis,
        #                    '/home/ec2-user/model/error_analysis_better_than_weight_test.txt', 'weight_test', is_better=True)
        # generate_error_txt(test_data, hypotheses_id, weights_id, reference, hypothesis,
        #                    '/home/ec2-user/model/error_analysis_worse_than_weight_test.txt', 'weight_test', is_better=False)

    return wer_score


def generate_error_txt(test_data, pred_id, compare_id, rewrite_id, reference, hypothesis, file_path, compare_type, all_hypotheses, all_wers, is_better=True):
    """Generate error files"""

    nbest_selected = test_data.loc[pred_id]
    nbest_selected_compare = test_data.loc[compare_id]
    rewrite_selected = test_data.loc[rewrite_id]

    with open(file_path, 'w') as f:
        count = 0
        errs = [0] * 3
        errs_compare = [0] * 3
        utt_id_good = []
        utt_id_bad = []
        for i in range(len(nbest_selected_compare)):
            better = nbest_selected.iloc[i]['WER'] if is_better else nbest_selected_compare.iloc[i]['WER']
            poorer = nbest_selected_compare.iloc[i]['WER'] if is_better else nbest_selected.iloc[i]['WER']
            if better < poorer:
            # if pred_id[i] == rewrite_id[i] and compare_id[i] != rewrite_id[i]:
                f.write('{:10}: {}\n'.format('Utt_id', nbest_selected.iloc[i]['utt_id']))
                f.write('{:10}: {}\n'.format('True', reference[pred_id[i]]))
                f.write('{:10}: {}\n'.format('Model', hypothesis[pred_id[i]]))
                f.write('{:10}: {}\n'.format(compare_type, hypothesis[compare_id[i]]))
                f.write('{:10}: {}\n'.format('Model WER', nbest_selected.iloc[i]['WER']))
                f.write('{:10}: {}\n'.format(compare_type+'WER', nbest_selected_compare.iloc[i]['WER']))
                f.write('{:10}: {:5}, {:5}, {:5}\n'.format('Model Sub/Ins/Del/',
                                                           nbest_selected.iloc[i]['#subErr'],
                                                           nbest_selected.iloc[i]['#insErr'],
                                                           nbest_selected.iloc[i]['#delErr']))
                f.write('{:10}: {:5}, {:5}, {:5}\n'.format(compare_type+' Sub/Ins/Del/',
                                                           nbest_selected_compare.iloc[i]['#subErr'],
                                                           nbest_selected_compare.iloc[i]['#insErr'],
                                                           nbest_selected_compare.iloc[i]['#delErr']))
                f.write('---- hypothesis ----\n')
                for ii, jj in zip(all_hypotheses[i], all_wers[i]):
                    f.write(' {:20}| wer: {:.3f} \n'.format(ii, jj))
                f.write('-------------------\n\n')
                count += 1

                for j, col in enumerate(['#subErr', '#insErr', '#delErr']):
                    errs[j] += nbest_selected.iloc[i][col]
                    errs_compare[j] += nbest_selected_compare.iloc[i][col]
                # if rewrite_selected.iloc[i]['WER'] < nbest_selected_compare.iloc[i]['WER']:
                #     utt_id_good.append(nbest_selected.iloc[i]['utt_id'])
                # else:
                #     utt_id_bad.append(nbest_selected.iloc[i]['utt_id'])


        # Total counts
        f.write('\n')
        f.write('{:10}: {:5}, {:5}, {:5}\n'.format('Model Sub/Ins/Del/', errs[0], errs[1], errs[2]))
        f.write('{:10}: {:5}, {:5}, {:5}\n'.format(compare_type + ' Sub/Ins/Del/', errs_compare[0], errs_compare[1], errs_compare[2]))
        f.write('\n')

        if is_better:
            f.write('Number of model counts better than {}: {}'.format(compare_type, count))
            print('Number of model counts better than {}: {}'.format(compare_type, count))
        else:
            f.write('Number of model counts poorer than {}: {}'.format(compare_type, count))
            print('Number of model counts poorer than {}: {}'.format(compare_type, count))

        # results = {'utt_id_good': utt_id_good, 'utt_id_bad': utt_id_bad}
        # import pickle
        # with open('/home/ec2-user/model/id_lists.pkl', 'wb') as f:
        #     pickle.dump(results, f)


def get_best_id(scores, a, b):
    """ Get the best nbest id based on scores
    :param scores: all rank scores for nbest
    :param a: start index of a group
    :param b: end index of a group
    :return: real_id: best nbest index of a group
    """
    predicted_sorted_indexes = np.argsort(scores[a:b])[::-1]  # from highest to lowest
    best_id = predicted_sorted_indexes[0]
    real_id = list(range(a, b))[best_id]
    return real_id

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


def calculate_ndcg(qids, targets, preds, k):
    """Predicts the scores for the test dataset and calculates the NDCG value.
    :param qids: numpy array of utterance id
    :param targets: target rank
    :param preds: predicted rank
    :param k: int
        This is used to compute the NDCG@k
    :return: average_ndcg: float
        This is the average NDCG value of all the queries
    :return: every_qid_ndcg: Numpy array of scores
        This contains an array or the predicted scores for each nbest.
    """
    query_groups = get_groups(qids)  # (qid,from,to)
    all_ndcg = []
    every_qid_ndcg = collections.OrderedDict()

    for qid, a, b in query_groups:
        predicted_sorted_indexes = np.argsort(preds[a:b])[::-1] # from highest to lowest
        t_results = targets[a:b] # target ranks
        t_results = t_results[predicted_sorted_indexes]

        dcg_val = dcg_k(t_results, k)
        idcg_val = ideal_dcg_k(t_results, k)
        ndcg_val = (dcg_val / idcg_val)
        all_ndcg.append(ndcg_val)
        every_qid_ndcg.setdefault(qid, ndcg_val)

    average_ndcg = np.nanmean(all_ndcg)
    return average_ndcg, every_qid_ndcg


def get_groups(qids):
    """Makes an iterator of query groups on the provided list of query ids.
    :param qids: array_like of shape = [n_samples]
        List of query ids.
    :return: row : (qid, int, int)
        Tuple of query id, from, to.
    """
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


def dcg_k(scores, k):
    """
    Returns the DCG value of the list of scores and truncates to k values.
    :param scores: list
        Contains labels in a certain ranked order
    :param k: int
        In the amount of values you want to only look at for computing DCG
    :return: DCG_val: int
        This is the value of the DCG on the given scores
    """
    return np.sum([(np.power(2, scores[i]) - 1) / np.log2(i + 2)
                    for i in range(len(scores[:k]))])


def ideal_dcg_k(scores, k):
    """
    Returns the Ideal DCG value of the list of scores and truncates to k values.
    :param scores: list
            Contains labels in a certain ranked order
    :param k: int
            In the amount of values you want to only look at for computing DCG
    :return: Ideal_DCG_val: int
            This is the value of the Ideal DCG on the given scores
    """
    scores = [score for score in sorted(scores)[::-1]]
    return dcg_k(scores, k)


def calculate_wer(reference, hypothesis):
    """
    Calculation of WER with Levenshtein distance.
    Works only for iterables up to 254 elements (uint8).
    O(nm) time and space complexity.
    :param reference: reference text
    :param hypothesis: hypothesis text
    :return: WER score
    """
    print('Calculating WER...')
    reference = reference.split()
    hypothesis = hypothesis.split()

    # initialisation
    d = np.zeros((len(reference) + 1) * (len(hypothesis) + 1), dtype=np.uint8)
    d = d.reshape((len(reference) + 1, len(hypothesis) + 1))
    for i in range(len(reference) + 1):
        for j in range(len(hypothesis) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(reference)][len(hypothesis)] / float(len(reference))

def tokenize(sent, dataset, vocab, is_hyp):
    """Tokenize the texts based on vocab"""

    if is_hyp:
        sent = dataset.remove_wake_words(sent)

    # tokenize
    sent = re.sub(re.compile(r'[/(){}\[\]\|@,;]'), ' ', sent)  # replace REPLACE_BY_SPACE_RE symbols by space in sent
    sent = re.sub(re.compile(r'[^0-9a-z #+_]'), '', sent)  # delete symbols which are in BAD_SYMBOLS_RE from sent
    sent = re.sub(r"[ ]+", " ", sent)
    sent = re.sub(r"\!+", "!", sent)
    sent = re.sub(r"\,+", ",", sent)
    sent = re.sub(r"\?+", "?", sent)

    token_ids = []
    for word in sent.split(' '):
        if word in vocab:
            token_ids.append(vocab[word])
        else:
            token_ids.append(vocab['[UNK]'])
    return token_ids

