import os

class Config:
        """
        Feature set
        1. score:                           scores from ESPNet after running Librispeech 1-pass
        2. length:                          "hyp_length"
        3. asr confidence score:            "bce_score", "bce_mwer_score", "bce_score_pointwise", "bce_mwer_score_pointwise"
        """
        FEATURE_to_train = ["score", "hyp_length"] # features to train bert listwise confidence model
        FEATURE_public = ["score", "hyp_length", "bce_score", "bce_mwer_score"] # features to train & evaluate lambdamart model

        ####################################
        # File paths
        train_path = './data/libri_subset/train-all-test.csv'
        test_path = './data/libri_subset/test-clean-test.csv'

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
        epochs = 300

        hidden_dim = 2560
        attn_head = 8
        bert_dim = 768

        # Learning rate guide
        ########### transformer rescorer ###########
        # use batch_size 128
        # bert
        # ce: 5e-5
        # mwer: 5e-5

        # mulan
        # ce: 5e-6
        # mwer: 5e-6

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

        # mulan
        # bert_dim: 1024
        # rescorer: 5e-6
        # mse loss: 1e-6
        # bce loss: 1e-6
        # bce loss2: 1e-6
        # ce loss: 1e-6
        # margin loss: 5e-6

        ########### bert ltr ###########
        # feature only: epoch 300, batch_size 128, lr 5e-5
        # listwise: epoch 300, batch_size 128, lr 5e-7

opt = Config()





