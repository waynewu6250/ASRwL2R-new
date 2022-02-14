import lightgbm as lgb
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score
from utils import *
import os
import shap
import pickle
import matplotlib.pyplot as plt
from config import opt
import torch
from tqdm import tqdm
import time
import scipy

from bert_data_ltr import CoreDataset, get_dataloader
from model.bert_model import BertEmbedding
from model.bert_ltr_rescorer import BertLTRRescorer

class Predictor:

    def __init__(self, args, opt, test_dataset, features):
        """Class: Train a model with specified configurations."""

        self.args = args
        self.opt = opt
        self.test_dataset = test_dataset
        self.utt_ids = test_dataset.data.utt_id
        self.dtest = test_dataset.data_for_use
        self.dtest_y = test_dataset.label
        self.dgroup_test = test_dataset.group

        self.baseline_data = self.dtest #self.dtest[features]
        self.feature_num = opt.feature_num_test
        self.test_data = test_dataset.data

        # Add reference and hypothesis to test_data
        # Total columns for test_data:
        # utt_id, group_id, 8 features, snr, WER, rank, 3 WER-related values, truth_from_file, truth
        self.true_words = {}

        self.lr_path = './checkpoints/{}.pkl'.format(self.args.model)
        self.model_path = './checkpoints/{}{}.mod'.format(self.args.model, self.args.postfix)

        # gpu
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch.backends.cudnn.enabled = False

        # bert ltr
        # self.pretrain_embed = self.args.embed
        # self.checkpoint_path = args.checkpoint_path
        # self.model = BertEmbedding(self.checkpoint_path)
        # self.model = self.model.to(self.device)
        # if os.path.exists('./checkpoints/model_{}.pth'.format(self.pretrain_embed)):
        #     self.model.load_state_dict(
        #         torch.load('./checkpoints/model_{}.pth'.format(self.pretrain_embed), map_location='cpu'))
        #     print('Pretrained bert model has been loaded...')


    def predict(self, file_type):
        """Predict scores based on the chosen trained model"""

        # Calculate oracle WER
        print('-------------------')
        print('Oracle WER scores: ')
        calculate_best_wer(self.utt_ids,
                           -self.test_data.WER.to_numpy().astype(np.float32),
                           self.test_data.truth.to_numpy(),
                           self.test_data.hyp.to_numpy(),
                           self.test_data)
        print('-------------------')

        # Calculate 1-best WER
        print('1-best WER scores: ')
        calculate_1best_wer(self.utt_ids,
                            self.test_data.truth.to_numpy(),
                            self.test_data.hyp.to_numpy(),
                            self.test_data)
        print('-------------------')

        # Calculate predicted model's WER
        print('Predicted scores for {}: '.format(self.args.model))
        if self.args.model == 'fixed_weight':

            weight = pd.DataFrame(pd.Series([0.4, 0.25, 0.25, 0.75], index=self.baseline_data.columns, name=0))
            self.test_data['score'] = -self.baseline_data.dot(weight)

            rank_score = self.test_data.score.to_numpy()

        elif self.args.model == 'reg':

            if not os.path.exists(self.lr_path):
                raise RuntimeError('Failed to load pretrained model')

            lr = joblib.load(self.lr_path)
            pred_lr = lr.predict(self.dtest.values)
            print('Test R-squared for linreg is %.4f' % r2_score(self.dtest_y, pred_lr))

            rank_score = pred_lr

        elif self.args.model == 'lambdamart' or self.args.model == 'lambdamart_tune':

            if not os.path.exists(self.model_path):
                raise RuntimeError('Failed to load pretrained model')

            gbm = lgb.Booster(model_file=self.model_path)

            pred_y = gbm.predict(self.dtest)

            # Calculate ndcg
            print("all uid average ndcg@2: ", calculate_ndcg(self.utt_ids, self.dtest_y.values, pred_y, 2)[0])
            print("all uid average ndcg@3: ", calculate_ndcg(self.utt_ids, self.dtest_y.values, pred_y, 3)[0])
            print("all uid average ndcg@5: ", calculate_ndcg(self.utt_ids, self.dtest_y.values, pred_y, 5)[0])

            rank_score = pred_y

            # weight = pd.DataFrame(pd.Series([0.4, 0.25, 0.25, 0.75], index=self.baseline_data.columns, name=0))
            # self.test_data['score'] = -self.baseline_data.dot(weight)
            # rank_score_weight = self.test_data.score.to_numpy()

        elif self.args.model == 'bert_ltr':

            test_dataloader = get_dataloader(self.test_dataset, utterance_mode='group',
                                            pretrain_embed=self.pretrain_embed)

            model = BertLTRRescorer(self.device, self.pretrain_embed, self.checkpoint_path,
                                    test_dataloader.dataset.tokenizer.vocab_size, opt,
                                    pretrain_model=self.model)
            model_path = './bert_ltr_rescorer_{}_listwise.pth'.format(self.pretrain_embed)
            model = model.to(self.device)

            model.load_state_dict(torch.load(model_path))
            print('Pretrained bert ltr model has been loaded...')

            pred_y = []
            model.eval()
            for i, batch in enumerate(tqdm(test_dataloader,
                                           total=len(test_dataloader),
                                           desc='Batches',
                                           unit=' batches',
                                           ncols=80)):
                with torch.no_grad():
                    preds = model(batch)
                    _, _, batch_preds_clean = model.learning_to_rank_loss(preds, batch)

                pred_y += batch_preds_clean

            rank_score = pred_y

            weight = pd.DataFrame(pd.Series([0.4, 0.25, 0.25, 0.75], index=self.baseline_data.columns, name=0))
            self.test_data['score'] = -self.baseline_data.dot(weight)
            rank_score_weight = self.test_data.score.to_numpy()

        else:
            raise ValueError('Input valid model input!!')


        if file_type == 'train' or self.args.model == 'fixed_weight' or self.args.model == 'reg':
            # Original calculation
            listwise_bce = self.test_data.bce_score.to_numpy().astype(np.float32)
            calculate_best_wer(self.utt_ids,
                               listwise_bce,
                               self.test_data.truth.to_numpy(),
                               self.test_data.hyp.to_numpy(),
                               self.test_data,
                               run_error=self.args.run_error)
        elif file_type == 'fixed_weight':
            # Calculate WER based on linear weighting and scores
            alpha = [0, 0.0001, 0.0004, 0.0005, 0.008, 0.01, 0.03, 0.05, 0.06, 0.07, 0.08, 0.1, 0.3, 0.5, 0.6, 0.7]
            # alpha = [0.0078, 0.0079, 0.008, 0.0081, 0.0082]
            best_score = 1
            for a in alpha:
                # Uncomment the following line to use other score
                # listwise_bce = self.test_data.asr_conf_bert_bce_listwise.to_numpy().astype(np.float32)
                # listwise_bce_mwer = self.test_data.asr_conf_bert_bce_mwer_listwise.to_numpy().astype(np.float32)
                # mulan_mlm_score = 0.01 * self.test_data['mulan_distillbert_mlm_score'].values
                rank_score = rank_score_weight* a + np.array(pred_y)
                wer_score = calculate_best_wer(self.utt_ids,
                                               rank_score,
                                               self.test_data.truth.to_numpy(),
                                               self.test_data.hyp.to_numpy(),
                                               self.test_data,
                                               run_error=self.args.run_error)
                if wer_score < best_score:
                    best_score = wer_score
                    best_alpha = a
            print('Best WER score from wer file with {}: {:.5f}'.format(best_alpha, best_score))

        else:
            optim_func = self.optim_func_generator(pred_y)
            bound=np.asarray([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 20.0]])
            optim_result = scipy.optimize.dual_annealing(optim_func,
                                                         bound,
                                                         local_search_options={},
                                                         no_local_search=True,
                                                         initial_temp=5.e4,
                                                         callback=lambda x, f, context: print(x, f, context),
                                                         maxiter=1000, visit=2.95, accept=-5)
            weights = optim_result.x
            rank_score = self.baseline_data.dot(weights[:self.feature_num]).to_numpy() + weights[self.feature_num] * pred_y
            calculate_best_wer(self.utt_ids,
                               rank_score,
                               self.test_data.truth.to_numpy(),
                               self.test_data.hyp.to_numpy(),
                               self.test_data,
                               run_error=self.args.run_error)

        # Uncomment the following lines to compare two scores
        # rank_score_weight = rank_score_weight * 0.1 + np.array(pred_y)
        # calculate_best_wer(self.utt_ids,
        #                    rank_score,
        #                    rank_score_weight,
        #                    self.test_data.truth.to_numpy(),
        #                    self.test_data.hyp.to_numpy(),
        #                    self.test_data,
        #                    run_error=self.args.run_error)

    def optim_func_generator(self, pred_y):
        def tmp_func(weights):
            rank_score = self.baseline_data.dot(weights[:self.feature_num]).to_numpy() + weights[self.feature_num]*pred_y
            return calculate_min_wer(self.utt_ids, rank_score, self.test_data)
        return tmp_func


    def visualize(self, X):
        """Analyze LambdaMART model with feature importances & tree structures"""

        if not os.path.exists(self.model_path):
            raise RuntimeError('Failed to load pretrained model')

        gbm = lgb.Booster(model_file=self.model_path)

        if self.args.vis == 'plot_tree':
            """Visualize the decision process via tree path"""
            graph = lgb.create_tree_digraph(gbm, tree_index=5, name='tree' + str(2))
            graph.render(filename='./model/tree_plot_best', view=True)

        elif self.args.vis == 'plot_feature_importance':
            """Feature importance analysis"""

            importances = gbm.feature_importance(importance_type='gain')
            feature_names = gbm.feature_name()

            print('{:50} : {:10} : {}'.format('Feature Name', 'Importance', 'Percentage'))
            for feature_name, importance in zip(feature_names, importances):
                if importance != 0:
                    print('{:50} : {:10} : {:.4f}%'.format(feature_name, importance, importance / sum(importances) * 100))

            lgb.plot_importance(gbm, importance_type='gain')
            plt.savefig('./model/feature_importance_best.jpg', dpi=300, bbox_inches="tight")

        elif self.args.vis == 'plot_print_feature_shap':
            """SHAP Analysis for features"""

            # load JS visualization code to notebook
            shap.initjs()

            gbm.params["objective"] = "lambdarank"
            explainer = shap.TreeExplainer(gbm)
            shap_values = explainer.shap_values(X)

            # Feature analysis

            # 1. Feature to X's absolute mean importance (different from feature_importance)
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            plt.savefig('./model/shap/feature_best.jpg', dpi=300, bbox_inches="tight")

            # 2. Feature overall
            shap.summary_plot(shap_values, X, show=False)
            plt.savefig('./model/shap/feature_bar_best.jpg', dpi=300, bbox_inches="tight")

            # 3. Dependency plot for one features
            shap.dependence_plot('nlm', shap_values, X, interaction_index=None, show=False)
            plt.savefig('./model/shap/dependency_plot_best.jpg', dpi=300, bbox_inches="tight")

            # 4. Dependency plot for two features
            shap.dependence_plot('nlm', shap_values, X, interaction_index='acoustic_model', show=False)
            plt.savefig('./model/shap/dependency_plot2_best.jpg', dpi=300, bbox_inches="tight")

            # 5. Multivariate analysis
            # data_to_display = X[['nlm','neg_directedness_stable','neg_directedness','neg_confidence_stable']]
            # shap_interaction_values = explainer.shap_interaction_values(data_to_display)
            # shap.summary_plot(shap_interaction_values, data_to_display, max_display=4, show=False)
            # plt.savefig('./model/shap/multivariate.jpg', dpi=300, bbox_inches="tight")

            # 6. One data analysis
            shap.force_plot(explainer.expected_value, shap_values[10, :], X.iloc[10, :], show=False, matplotlib=True).savefig('./model/shap/single_data_best.jpg')

        elif self.args.vis == 'wer_plot':
            """Plot WER score variation during LambdaMART parameter tuning"""

            with open('./model/wer_scores_dev.pkl', 'rb') as f:
                results = pickle.load(f)

            plt.figure(figsize=(20, 20))
            for i in range(len(results)):
                plt.subplot(4, 3, i + 1)
                feature = list(results[i]['distributions'].keys())[0]
                y = results[i]['wer_scores']
                plt.plot(results[i]['distributions'][feature], y)
                plt.xlabel(feature)
                plt.ylabel('wer')
                plt.ylim([0.095, 0.0975])
                plt.title(feature)
            plt.savefig('./model/wer/dev.jpg')

        print('Completed downloading images.')









