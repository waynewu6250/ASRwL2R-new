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
import pandas as pd
import json
import csv
import pickle
import os
from collections import defaultdict
from scipy.stats import rankdata
import kaldi_io
import jiwer
import gzip
import glob

class L2RDataset:

    def __init__(self, path, prefix, features, opt, preload=False, feature_num=8):
        """A class object to store data
        :param paths: a dictionary of file paths
               keys: 'csv_path': final processed csv file
        :param prefix: file path header
        :param features: feature list to use
        :param opt: configurations defined in config.py
        :param preload: whether to use the processed data or preprocess again
        :param nbest_all: whether to use the utterances with only one hypothesis
        :param nbest_extend: whether to extend nbest with rewrite
        :param feature_num: number of ASR features
        """

        self.opt = opt
        self.feature_num = feature_num
        self.audio_dic = {}

        if preload:
            self.data = pd.read_csv(path)
            self.data["truth"] = self.data["truth"].replace(np.NaN, "") # Some values are missing in WER truth
            self.data["hyp"] = self.data["hyp"].map(lambda x: str(x)) # Make sure texts are all strings, not float
        else:
            raise ValueError('Please parse the data first to generate csv files!!!')

        self.data_for_use = self.data[features]
        self.label = self.data['rank']
        self.group = self.get_group()
        self.num_utterances = self.data.loc[len(self.data_for_use)-1]['group_id']+1

    def get_groups(self, qids):
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

    def get_group(self):
        """Get group size"""

        group = []
        prev_idx = 0
        count = 0
        for idx in self.data.group_id:
            if prev_idx != idx:
                group.append(count)
                count = 0
            count += 1
            prev_idx = idx
        group.append(count)
        return np.array(group)

    @staticmethod
    def get_wer_rank(nbest_wer):
        """Get WER ranking (LambdaMART can only have integer label)
           Score from 0 to 4
        """
        # Uncomment lines below to generate rank targets for regression model
        # def classify(x):
        #     if x == 1:
        #         return 4
        #     elif 0.8 <= x < 1:
        #         return 3
        #     elif 0.714 <= x < 0.8:
        #         return 2
        #     elif 0.5 <= x < 0.714:
        #         return 1
        #     else:
        #         return 0
        # scores = [classify(1-float(nbest['WER'])) for nbest in nbest_wer]
        # return scores

        scores = [1-float(nbest['WER']) for nbest in nbest_wer]
        return rankdata(scores, method='max')

        # Uncomment lines below to change the way to generate rank
        # x = [1-float(nbest['WER']) for nbest in nbest_wer]
        # idx = np.argsort(x)[::-1]
        #
        # rank = 4
        # prev_val = x[idx[0]]
        # ranks = [0] * len(idx)
        # for i in range(len(idx)):
        #     if x[idx[i]] != prev_val:
        #         rank -= 1
        #     ranks[idx[i]] = rank
        #     prev_val = x[idx[i]]
        #
        # return ranks
