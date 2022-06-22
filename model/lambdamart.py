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
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, GroupKFold
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelBinarizer
import time
import itertools
import pickle
import copy
import random
import math

class LGBMOpt:

    def __init__(self, data_train, data_valid, X, y, X_val, y_val, group, dev_data):
        """Hyper-parameter optimization pipeline for lightgbm with lambamart."""

        self.data_train = data_train
        self.data_valid = data_valid
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.group = group
        self.dev_data = dev_data

        self._params_cv = {'cv_folds': 5,
                           'early_stopping_rounds': 50,
                           'stratified': False,
                           'iid': False,
                           'metrics': 'ndcg'}

        # Default parameters for lightgbm
        self._params = {'boosting_type': 'gbdt',
                        'objective': 'lambdarank',
                        'ndcg_eval_at': [1, 3, 5],
                        'feature_pre_filter': False,

                        'learning_rate': 0.1,
                        'max_depth': 5,
                        'n_estimators': int(1e3),
                        'num_leaves': 50,  # num_leaves<=2**max_depth-1
                        'min_data_in_leaf': 5,
                        'min_sum_hessian_in_leaf': 50,
                        'subsample': 0.6,
                        'min_split_gain': 1,
                        'subsample_freq': 1,
                        'feature_extraction': 0,
                        'colsample_bytree': 0.6,
                        'reg_alpha': 0.01,
                        'reg_lambda': 1,
                        'num_threads': 16,
                        'verbose': -1}

        # All parameters to be tuned
        # original tune range
        # self._tune_range = {'max_depth': list(range(3, 8, 2)),
        #                     'num_leaves': list(range(50, 170, 30)),  # 2**max_depth-1 [7, 31, 127, 511]
        #                     'min_data_in_leaf': [5, 20, 50, 100],
        #                     'min_sum_hessian_in_leaf': [10, 50, 100, 200, 500],
        #                     'min_split_gain': [i / 10. for i in range(0, 4)],
        #                     'feature_extraction': [i / 10. for i in range(0, 4)],
        #                     'subsample': [i / 10.0 for i in range(6, 10)],
        #                     'subsample_freq': [1, 5, 10],
        #                     'colsample_bytree': [i / 10.0 for i in range(6, 10)],
        #                     'reg_alpha': [0.01, 0.05, 0.1, 0.2],  # [1e-6, 1e-2, 0.1, 1, 100]
        #                     'reg_lambda': [80, 100, 120, 140]
        #                     }

        # test tune range
        self._tune_range = {'max_depth': list(range(1, 20, 2)),
                            'num_leaves': list(range(10, 50, 5)),  # 2**max_depth-1 [7, 31, 127, 511]
                            'min_data_in_leaf': list(range(5, 50, 5)),
                            'min_sum_hessian_in_leaf': list(range(5, 100, 5)),
                            'min_split_gain': [i / 10. for i in range(0, 8)],
                            'feature_extraction': [i / 10. for i in range(0, 10)],
                            'subsample': [i / 10.0 for i in range(2, 10)],
                            'subsample_freq': [0, 1, 2, 5, 8, 10, 20],
                            'colsample_bytree': [i / 10.0 for i in range(2, 10)],
                            'reg_alpha': [0, 0.01, 0.05, 0.1, 0.2],  # [1e-6, 1e-2, 0.1, 1, 100]
                            'reg_lambda': [0, 0.1, 1, 10, 100, 150, 200]
                            }

        # For each step, tune different parameters
        self._params_tune = [['n_estimators'],
                             ['max_depth'],
                             ['num_leaves'],
                             ['min_data_in_leaf'],
                             ['min_sum_hessian_in_leaf'],
                             ['feature_extraction'],
                             ['subsample'],
                             ['subsample_freq'],
                             ['colsample_bytree'],
                             ['min_split_gain'],
                             ['reg_alpha'],
                             ['reg_lambda']]
                             # ['n_estimators'],
                             # ['max_depth', 'num_leaves', 'min_data_in_leaf', 'min_sum_hessian_in_leaf'],
                             # ['feature_extraction', 'subsample', 'subsample_freq', 'colsample_bytree'],
                             # ['min_split_gain'],
                             # ['reg_alpha', 'reg_lambda']]

        self.best_score = 1 #-1e10
        self._temp_score = 1 #-1e10
        self._score_mult = 1
        self.max_rounds = 1
        self._step = 0
        self.best_model = None

    def tune_params(self):
        """Tune the LambdaMART parameters."""

        self._start_time = time.time()
        iround = 0
        all_results = []
        while iround < self.max_rounds:
            print('\nLearning rate for iteration %i: %f.' % (iround + 1, self._params['learning_rate']))
            self._step = 0
            while self._step < len(self._params_tune):
                istep_time = time.time()
                if self._step == 0:
                    # First find the optimal number of n estimators
                    self.get_n_estimators()
                else:
                    # Then perform grid search
                    results = self.grid_search_cv()
                    all_results.append(results)
                self.print_progress(istep_time, iround=iround,
                                    max_rounds=self.max_rounds)  # print params and performance
                self._step += 1
            iround += 1
        print('\nFinal best parameters: ')
        for k, v in self._params.items():
            print('{}: {}'.format(k, v))

        with open('./model/wer_scores.pkl', 'wb') as f:
            pickle.dump(all_results, f)


    def get_n_estimators(self):
        """Get initial n estimator estimation for later optimization."""

        kwargs_cv = {'num_boost_round': self._params['n_estimators'],
                     'nfold': self._params_cv['cv_folds'],
                     'early_stopping_rounds': self._params_cv['early_stopping_rounds'],
                     'stratified': self._params_cv['stratified'],
                     'metrics': self._params_cv['metrics'],
                     'shuffle': False,
                     'verbose_eval': 50,
                     'show_stdv': True,
                     'seed': 0}
        cv_results = lgb.cv(self._params, self.data_train, **kwargs_cv)

        print('best n_estimators:', len(cv_results[kwargs_cv['metrics']+'@1-mean']))
        print('best cv score:', cv_results[kwargs_cv['metrics']+'@1-mean'][-1])
        print()

        self._params['n_estimators'] = int(len(cv_results[kwargs_cv['metrics'] + '@1-mean'])
                                           / (1-1/self._params_cv['cv_folds']))

    def grid_search_cv(self):
        """Main grid search algorithm with cross validation.

        Based on the range of parameters, for the current setting,
        training data around ~20k nbest will take ~10min to find optimal parameters.
        """

        param_distributions = self.get_params_tune()
        perm_list = [v for k, v in param_distributions.items()]
        possible_combinations = list(itertools.product(*perm_list))

        temp_params = {k: v for k, v in self._params.items()}
        print('Running on parameters:')

        wer_scores_stored = []
        for comb in possible_combinations:
            for i, k in enumerate(param_distributions.keys()):
                print('{}: {}'.format(k, comb[i]))
                temp_params[k] = comb[i]

            kwargs_cv = {'num_boost_round': self._params['n_estimators'],
                         'nfold': self._params_cv['cv_folds'],
                         'early_stopping_rounds': self._params_cv['early_stopping_rounds'],
                         'stratified': self._params_cv['stratified'],
                         'metrics': self._params_cv['metrics'],
                         'shuffle': False,
                         'verbose_eval': -1,
                         'show_stdv': True,
                         'seed': 0}
            cv_results = lgb.cv(temp_params, self.data_train, **kwargs_cv, return_cvbooster=True)
            best_cv_scores = cv_results[kwargs_cv['metrics'] + '@1-mean'][-1]
            best_booster = cv_results['cvbooster'].boosters[-1]
            for idx in [1, 3, 5]:
                print('cv_agg ndcg@{}: {}'.format(idx, cv_results[kwargs_cv['metrics'] + '@{}-mean'.format(idx)][-1]))

            y_pred = best_booster.predict(self.X_val)
            best_wer_scores = self.calculate_best_wer(self.dev_data.utt_id,
                                                      y_pred,
                                                      self.dev_data.truth.to_numpy(),
                                                      self.dev_data.hyp.to_numpy(),
                                                      self.dev_data)
            wer_scores_stored.append(best_wer_scores)
            print('cv_agg wer: {}'.format(best_wer_scores))
            print()

            # update best model if cv_score is improved
            # if (best_cv_scores * self._score_mult) > (self.best_score * self._score_mult):
            #
            #     self.best_score = best_cv_scores
            #     self.best_model = best_booster
            #
            #     # update tuned parameters with optimal values
            #     for key, value in temp_params.items():
            #         self._params[key] = value
            #     self._temp_score = best_cv_scores

            # update best model if wer_score is improved
            if best_wer_scores < self.best_score:

                self.best_score = best_wer_scores
                self.best_model = copy.deepcopy(best_booster)

                # update tuned parameters with optimal values
                for key, value in temp_params.items():
                    self._params[key] = value
                self._temp_score = best_wer_scores

        results = {'combinations': possible_combinations,
                   'distributions': param_distributions,
                   'wer_scores': wer_scores_stored}
        return results


    def get_params_tune(self):
        """Return dict of parameters to be tuned with values."""

        params_tune = self._params_tune[self._step]
        params_tune_dict = {}
        for key, value in self._tune_range.items():
            if key in params_tune:
                params_tune_dict[key] = value
        return params_tune_dict


    def print_progress(self, step_time, iround=None, max_rounds=None):
        """Print update on tuning progress."""

        if iround is not None and max_rounds is not None:
            print('\nIteration %i/%i.' % (iround + 1, max_rounds))
        else:
            print('\nIteration %i/%i.' % (1, 1))
        print('...Step %i: tune %s.' % (self._step, self._params_tune[self._step]))
        print('...Time elapsed for this iteration: %f min.' % ((time.time() - step_time) / 60))
        print('...Total time elapsed: %f min.' % ((time.time() - self._start_time) / 60))
        for item in self._params_tune[self._step]:
            print('...Optimal value %s: %f.' % (item, self._params[item]))
        print('...Model score: %f.' % (self._temp_score))
        print()

    def get_groups(self, qids):
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

    def calculate_best_wer(self, qids, preds, reference, hypothesis, test_data):
        """Predicts the scores for the test dataset and calculates the best WER value based on ranking."""

        query_groups = self.get_groups(qids)  # (qid,from,to)
        references = []
        hypotheses = []
        hypotheses_id = []
        for qid, a, b in query_groups:
            predicted_sorted_indexes = np.argsort(preds[a:b])[::-1]  # from highest to lowest
            best_id = predicted_sorted_indexes[0]
            real_id = list(range(a, b))[best_id]
            references.append(str(reference[real_id]))
            hypotheses.append(str(hypothesis[real_id]))
            hypotheses_id.append(real_id)

        # Calculate from wer file
        nbest_selected = test_data.loc[hypotheses_id]
        nbest_selected['total_error'] = nbest_selected['#subErr'].astype('float64') + nbest_selected['#insErr'].astype(
            'float64') + nbest_selected['#delErr'].astype('float64')
        wer_score = sum(nbest_selected['total_error']) / sum(nbest_selected['#refWord'].astype('float64'))
        return wer_score




########################################################################
    # Depreciated (under construction/save here for future reference)

    def ndcg_score(self, ground_truth, predictions, k=5):
        """Normalized discounted cumulative gain (NDCG) at rank K."""

        lb = LabelBinarizer()
        lb.fit(range(len(predictions) + 1))
        T = lb.transform(ground_truth)

        scores = []

        # Iterate over each y_true and compute the DCG score
        for y_true, y_score in zip(T, predictions):
            actual = self.dcg_score(y_true, y_score, k)
            best = self.dcg_score(y_true, y_true, k)
            score = float(actual) / float(best)
            scores.append(score)

        return np.mean(scores)


    @staticmethod
    def dcg_score(y_true, y_score, k=5):
        """Discounted cumulative gain (DCG) at rank K."""

        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])

        gain = 2 ** y_true - 1

        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gain / discounts)

    def apply_gridsearch(self):
        """Apply grid search on ml algorithm to specified parameters
        returns updated best score and parameters"""

        model = lgb.LGBMRanker(**self._params)

        ndcg_scorer = make_scorer(self.ndcg_score, k=1)

        # group info
        group_info = self.group.astype(int)
        flatted_group = np.repeat(range(len(group_info)), repeats=group_info)
        gkf = GroupKFold(n_splits=3)
        cv = gkf.split(self.X, self.y, groups=flatted_group)
        cv_group = gkf.split(self.X, groups=flatted_group)  # separate CV generator for manual splitting groups

        # generator produces `group` argument for LGBMRanker for each fold
        def group_gen(flatted_group, cv):
            for train, _ in cv:
                yield np.unique(flatted_group[train], return_counts=True)[1]

        gen = group_gen(flatted_group, cv_group)
        gsearch = RandomizedSearchCV(estimator=model, param_distributions=self.get_params_tune(),
                                     n_iter=10,
                                     verbose=2,
                                     cv=cv,
                                     refit=False,
                                     scoring=ndcg_scorer,
                                     n_jobs=self._params['n_jobs'])

        gsearch.fit(self.X, self.y, group=next(gen))

        # update best model if best_score is improved
        if (gsearch.best_score_ * self._score_mult) > (self.best_score * self._score_mult):
            self.best_model = clone(gsearch.best_estimator_)
            self.best_score = gsearch.best_score_

        # update tuned parameters with optimal values
        for key, value in gsearch.best_params_.items():
            self._params[key] = value
        self._temp_score = gsearch.best_score_



