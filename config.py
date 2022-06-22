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
import os

class Config:
        """
        Feature set
        1. score:                           scores from ESPNet after running Librispeech 1-pass
        2. length:                          "hyp_length"
        3. asr confidence score:            "bce_score", "bce_mwer_score", "bce_score_pointwise", "bce_mwer_score_pointwise"
        """
        # FEATURE_to_train = ["score", "hyp_length"] # features to train bert listwise confidence model
        FEATURE_to_train = ["hyp_length", "decoder", "ctc", "lm", "score"] # "score"
        # FEATURE_public = ["score", "hyp_length", "bce_score", "bce_mwer_score"] # features to train & evaluate lambdamart model
        FEATURE_public = ["hyp_length", "decoder", "ctc", "lm"]#, "bce_score", "bce_mwer_score"] # "score"
        feature_num_train = len(FEATURE_to_train)
        feature_num_test = len(FEATURE_public)

        ####################################
        # File paths
        # librispeech espnet
        train_path = './data/espnet_parsed/train-all.csv'
        test_path = './data/espnet_parsed/test_clean.csv'

        files_to_use = ['.scored_nbest_dataset',
                        '.snr_file.txt',
                        '.nbest_wer_csv.txt',
                        '_large.csv',
                        '.wscp',
                        '.mulan.csv',
                        '_audio_dic.pkl']
        paths = ['score_nbest_path',
                 'snr_path',
                 'wer_path',
                 'csv_path',
                 'wscp_path',
                 'distillbert_score_path',
                 'audio_dic_path'
        ]
        
        ########################################################################

        # parameters
        batch_size = 16
        learning_rate_bert = 1e-6
        learning_rate = 5e-7
        epochs = 15

        hidden_dim = 2560
        attn_head = 8
        bert_dim = 768

        # Learning rate guide
        ########### transformer rescorer ###########
        # use batch_size 128
        # bert
        # ce: 5e-5
        # mwer: 5e-5

        ########### bert rescorer ###########
        # use batch_size 128
        # bert
        # bert_dim: 768
        # rescorer: 5e-6
        # mse loss: 1e-6
        # bce loss: 1e-6
        # bce loss2: 1e-6
        # ce loss: 1e-6
        # margin loss: 5e-6

opt = Config()





