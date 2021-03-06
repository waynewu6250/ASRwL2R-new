import lightgbm as lgb
from sklearn.linear_model import LinearRegression
import joblib
import torch
from torch.optim import Adam
import os
from tqdm import tqdm
import numpy as np

from bert_data_ltr import CoreDataset, get_dataloader
from model.bert_model import BertEmbedding
from model.bert_ltr_rescorer import BertLTRRescorer
from model.lambdamart import LGBMOpt
from model.ptranking.ltr_adhoc.eval.eval_utils import calculate_best_wer
from config import opt

class Trainer:

    def __init__(self, args, train_dataset, dev_dataset):
        """Class: Train a model with specified configurations."""

        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.dtrain = train_dataset.data_for_use
        self.dvalid = dev_dataset.data_for_use
        self.dtrain_y = train_dataset.label
        self.dvalid_y = dev_dataset.label
        self.dgroup_train = train_dataset.group
        self.dgroup_valid = dev_dataset.group
        self.dev_data = dev_dataset.data
        self.data_train = lgb.Dataset(self.dtrain, self.dtrain_y, group=self.dgroup_train, free_raw_data=False)
        self.data_valid = lgb.Dataset(self.dvalid, self.dvalid_y, group=self.dgroup_valid, free_raw_data=False)
        print('Number of features used: ', len(self.dtrain.columns))

        self.data_train.set_group(self.dgroup_train)
        self.data_valid.set_group(self.dgroup_valid)

        self.lr_path = './checkpoints/{}.pkl'.format(self.args.model)
        self.model_path = './checkpoints/{}.mod'.format(self.args.model)

        # gpu
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch.backends.cudnn.enabled = False

        # bert ltr
        # self.pretrain_embed = self.args.embed
        # self.checkpoint_path = args.checkpoint_path
        # self.model = BertEmbedding(self.checkpoint_path)
        # self.model = self.model.to(self.device)
        # if os.path.exists('./model_{}.pth'.format(self.pretrain_embed)):
        #     self.model.load_state_dict(
        #         torch.load('./model_{}.pth'.format(self.pretrain_embed), map_location='cpu'))
        #     print('Pretrained bert model has been loaded...')


    def fit(self):
        """Main fitting pipeline"""

        if self.args.model == 'reg':
            print('Running linear regression...')

            self.lr = LinearRegression()
            self.lr.fit(self.dtrain, self.dtrain_y)
            print(self.lr.score(self.dtrain, self.dtrain_y))
            joblib.dump(self.lr, self.lr_path)

        elif self.args.model == 'mart':
            print('Running MART...')

            param = {
                'feature_fraction': 0.75,
                'metric': 'rmse',
                'nthread': 1,
                'min_data_in_leaf': 2**7,
                'bagging_fraction': 0.75,
                'learning_rate': 0.75,
                'objective': 'mse',
                'bagging_seed': 2**7,
                'num_leaves': 2**7,
                'bagging_freq': 1,
                'verbose': 2
            }

            self.bst = lgb.train(
                param, self.data_train, 100)
            self.bst.save_model(self.model_path)

        elif self.args.model == 'lambdamart':
            print('Running LambdaMART...')

            # original param
            # param = {
            #     "task": "train",
            #     "num_leaves": 50,
            #     "min_data_in_leaf": 0,
            #     "min_sum_hessian_in_leaf": 100,
            #     "objective": "lambdarank",
            #     "metric": "ndcg",
            #     "ndcg_eval_at": [1, 3, 5],
            #     "learning_rate": .1,
            #     "num_threads": 16
            # }

            # tune by ndcg
            # param = {'boosting_type': 'gbdt',
            #             'objective': 'lambdarank',
            #             'ndcg_eval_at': [1, 3, 5],
            #             'feature_pre_filter': False,
            #
            #             'learning_rate': 0.1,
            #             'max_depth': 3,
            #             'n_estimators': int(1e3),
            #             'num_leaves': 50,  # num_leaves<=2**max_depth-1
            #             'min_data_in_leaf': 5,
            #             'min_sum_hessian_in_leaf': 200,
            #             'subsample': 0.8,
            #             # 'subsample_freq': 5,
            #             'colsample_bytree': 0.8,
            #             # 'reg_alpha': 0,
            #             # 'reg_lambda': 1,
            #             'num_threads': 16,
            #             'verbose': -1}

            # best for small dataset
            # param  = {'boosting_type': 'gbdt',
            #           'objective': 'lambdarank',
            #           'ndcg_eval_at': [1, 3, 5],
            #           'feature_pre_filter': False,
            #
            #           'learning_rate': 0.1,
            #           'max_depth': 3,
            #           'n_estimators': 290,
            #           'num_leaves': 50,  # num_leaves<=2**max_depth-1
            #           'min_data_in_leaf': 5,
            #           'min_sum_hessian_in_leaf': 10,
            #           'subsample': 0.6,
            #           'min_split_gain': 1,
            #           'subsample_freq': 1,
            #           'feature_extraction': 0,
            #           'colsample_bytree': 0.6,
            #           'reg_alpha': 0.01,
            #           'reg_lambda': 1,
            #           'num_threads': 16,
            #           'verbose': -1}

            # best for large dataset
            param = {'boosting_type': 'gbdt',
                     'objective': 'lambdarank',
                     'ndcg_eval_at': [1, 3, 5],
                     'feature_pre_filter': False,

                     'learning_rate': 0.1,
                     'max_depth': 5,
                     'n_estimators': int(self.args.iterations),
                     'num_leaves': 50,  # num_leaves<=2**max_depth-1
                     'min_data_in_leaf': 5,
                     'min_sum_hessian_in_leaf': 50,
                     'subsample': 0.6,
                     'min_split_gain': 1,
                     'subsample_freq': 1,
                     'feature_extraction': 0,
                     'colsample_bytree': 0.6,
                     'reg_alpha': 0,
                     'reg_lambda': 1,
                     'num_threads': 16,
                     'verbose': -1}

            # test bert embed (0.01, 3000) mulan embed (0.005, 1860)
            # param = {'boosting_type': 'gbdt',
            #          'objective': 'lambdarank',
            #          'ndcg_eval_at': [1, 3, 5],
            #          'feature_pre_filter': False,
            #
            #          'learning_rate': 0.05,
            #          'max_depth': 3,
            #          'n_estimators': 520,
            #          'num_leaves': 50,  # num_leaves<=2**max_depth-1
            #          'min_data_in_leaf': 5,
            #          'min_sum_hessian_in_leaf': 10,
            #          'subsample': 0.6,
            #          'min_split_gain': 1,
            #          'subsample_freq': 1,
            #          'feature_extraction': 0,
            #          'colsample_bytree': 0.6,
            #          'reg_alpha': 0.01,
            #          'reg_lambda': 1,
            #          'num_threads': 16,
            #          'verbose': -1}

            # test for small train dataset
            # param = {'boosting_type': 'gbdt',
            #          'objective': 'lambdarank',
            #          'ndcg_eval_at': [1, 3, 5],
            #          'feature_pre_filter': False,
            #
            #          'learning_rate': 0.1,
            #          'max_depth': 3,
            #          'n_estimators': int(self.args.iterations),
            #          'num_leaves': 50,  # num_leaves<=2**max_depth-1
            #          'min_data_in_leaf': 5,
            #          'min_sum_hessian_in_leaf': 10,
            #          'subsample': 0.6,
            #          'min_split_gain': 1,
            #          'subsample_freq': 1,
            #          'feature_extraction': 0,
            #          'colsample_bytree': 0.6,
            #          'reg_alpha': 0.01,
            #          'reg_lambda': 1,
            #          'num_threads': 16,
            #          'verbose': -1}

            # test for large train dataset
            # param = {'boosting_type': 'gbdt',
            #          'objective': 'lambdarank',
            #          'ndcg_eval_at': [1, 3, 5],
            #          'feature_pre_filter': False,
            #
            #          'learning_rate': 0.1,
            #          'max_depth': 5,
            #          'n_estimators': int(self.args.iterations),
            #          'num_leaves': 50,  # num_leaves<=2**max_depth-1
            #          'min_data_in_leaf': 5,
            #          'min_sum_hessian_in_leaf': 50,
            #          'subsample': 0.6,
            #          'min_split_gain': 1,
            #          'subsample_freq': 1,
            #          'feature_extraction': 0,
            #          'colsample_bytree': 0.6,
            #          'reg_alpha': 0,
            #          'reg_lambda': 1,
            #          'num_threads': 16,
            #          'verbose': -1}

            res = {}
            self.bst = lgb.train(
                param, self.data_train,
                valid_sets=[self.data_valid],
                valid_names=["valid"],
                num_boost_round=1000,
                evals_result=res,
                verbose_eval=10)

            # save the best model
            idx = np.argmax(res['valid']['ndcg@1'])
            self.bst.save_model(self.model_path, num_iteration=idx)
            print('Save the best model on iterations: ', idx)
            print('Best ndcg@1 score: ', res['valid']['ndcg@1'][idx])


        elif self.args.model == 'lambdamart_tune':
            print('Running LambdaMART Tuning...')

            lgbopt = LGBMOpt(self.data_train, self.data_valid,
                             self.dtrain, self.dtrain_y,
                             self.dvalid, self.dvalid_y,
                             self.dgroup_train, self.dev_data)
            lgbopt.tune_params()
            lgbopt.best_model.save_model(self.model_path)

        elif self.args.model == 'bert_ltr':
            self.train_bert_ltr()

    def train_bert_ltr(self):
        """Train end-to-end BERT LTR model"""

        train_dataloader = get_dataloader(self.train_dataset, utterance_mode='group', pretrain_embed=self.pretrain_embed)
        val_dataloader = get_dataloader(self.dev_dataset, utterance_mode='group', pretrain_embed=self.pretrain_embed)

        model = BertLTRRescorer(self.device, self.pretrain_embed, self.checkpoint_path, train_dataloader.dataset.tokenizer.vocab_size, opt,
                                pretrain_model=self.model)
        model_path = './checkpoints/bert_rescorer_bert_bce_listwise.pth'.format(self.pretrain_embed)
        model = model.to(self.device)

        optimizer = Adam(model.parameters(), weight_decay=0.01, lr=opt.learning_rate)

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print('Pretrained bert ltr model has been loaded...')

        best_wer = 100

        # Start training
        for epoch in range(opt.epochs):
            print("====== epoch %d / %d: ======" % (epoch + 1, opt.epochs))

            # Training Phase
            total_train_loss = 0
            total_train_hyps = 0
            total_train_preds = []
            model.train()
            for i, batch in enumerate(tqdm(train_dataloader,
                                           total=len(train_dataloader),
                                           desc='Batches',
                                           unit=' batches',
                                           ncols=80)):

                optimizer.zero_grad()
                batch_preds = model(batch)
                train_loss, batch_hyps, batch_preds_clean = model.learning_to_rank_loss(batch_preds, batch)

                train_loss.backward()
                optimizer.step()

                total_train_loss += train_loss
                total_train_hyps += batch_hyps
                total_train_preds += batch_preds_clean

            average_train_loss = total_train_loss / total_train_hyps
            print('Average train loss: {:.4f} '.format(average_train_loss))

            wer_train_score = calculate_best_wer(self.dev_data.utt_id,
                                           total_train_preds,
                                           self.dev_data.truth.to_numpy(),
                                           self.dev_data.hyp.to_numpy(),
                                           self.dev_data)

            # Validation Phase
            total_val_loss = 0
            total_val_hyps = 0
            total_val_preds = []
            model.eval()
            for i, batch in enumerate(tqdm(val_dataloader,
                                           total=len(val_dataloader),
                                           desc='Batches',
                                           unit=' batches',
                                           ncols=80)):
                with torch.no_grad():
                    preds = model(batch)
                    val_loss, batch_hyps, batch_preds_clean = model.learning_to_rank_loss(preds, batch)

                total_val_hyps += batch_hyps
                total_val_loss += val_loss
                total_val_preds += batch_preds_clean

            average_val_loss = total_val_loss / total_val_hyps
            print('Average val loss: {:.4f} '.format(average_val_loss))

            wer_val_score = calculate_best_wer(self.dev_data.utt_id,
                                           total_val_preds,
                                           self.dev_data.truth.to_numpy(),
                                           self.dev_data.hyp.to_numpy(),
                                           self.dev_data)

            if wer_val_score < best_wer:
                print('saving with wer of {}'.format(wer_val_score),
                      'improved over previous {}'.format(best_wer))
                best_wer = wer_val_score
                torch.save(model.state_dict(),
                           '/home/ec2-user/bert_ltr_rescorer_{}_listwise.pth'.format(self.pretrain_embed))

        print()
        print('Best total wer: {:.4f}'.format(best_wer))

