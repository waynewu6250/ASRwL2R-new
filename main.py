"""Main running script"""
import lightgbm as lgb
from time import time
from data.dataset_public import L2RDataset
from trainer import Trainer
from predictor import Predictor
from utils import *
from config import opt
import os

def load_dataset(path, args):
    """Load dataset object"""
    # paths = {path: prefix + file for path, file in zip(opt.paths, opt.files_to_use)}
    switcher = {'feature_to_train': opt.FEATURE_to_train,
                'feature_public': opt.FEATURE_public}
    FEATURE = switcher[args.feature_to_use]
    dataset = L2RDataset(path, None, features=FEATURE, opt=opt, preload=args.preload, nbest_all=args.nbest_all, nbest_extend=args.nbest_extend)
    return dataset, FEATURE

def load_data(args):
    """Load train, dev, test dataset"""
    train_dataset, feat = load_dataset(opt.train_path, args)
    # dev_dataset, feat = load_dataset(opt.dev_path, args)
    test_dataset, feat = load_dataset(opt.test_path, args)
    # sanity check
    print('Number of used features: ', len(feat))
    return train_dataset, test_dataset

def train(args):
    """Main training pipeline"""

    start = time()

    ################ Loading data ################
    # Load ASR dataset
    print('Loading data...')
    train_dataset, test_dataset = load_data(args)
    # print(train_dataset.data.head(30))
    print('Number of utterances in training data: ', train_dataset.num_utterances)
    # print('Number of utterances in validation data: ', dev_dataset.num_utterances)
    print('Number of utterances in testing data: ', test_dataset.num_utterances)
    print('Number of total training data: ', len(train_dataset.data))
    # print('Number of total validation data: ', len(dev_dataset.data))
    print('Number of total testing data: ', len(test_dataset.data))

    if args.file == 'dev':
        dataset_to_validate = dev_dataset
    elif args.file == 'test':
        dataset_to_validate = test_dataset
    else:
        raise ValueError('Only input dev or test for validation')

    ################## Training ##################
    trainer = Trainer(args, train_dataset, dataset_to_validate)
    trainer.fit()

    print('Time used: ', time()-start)

def predict(args):
    """Main prediction pipeline"""

    ################ Loading data ################
    # Load ASR dataset
    print('Loading data...')
    train_dataset, test_dataset = load_data(args)

    if args.file == 'train':
        dataset_to_test = train_dataset
    elif args.file == 'dev':
        dataset_to_test = dev_dataset
    elif args.file == 'test':
        dataset_to_test = test_dataset

    ################ Predicting ##################
    predictor = Predictor(args, opt, dataset_to_test, features=opt.FEATURE_public)
    print('Number of utterances in data to test: ', dataset_to_test.num_utterances)
    print('Number of total data to test: ', len(dataset_to_test.data))
    predictor.predict(args.file)

def visualize(args):
    """Visualization of the current model"""
    ################ Loading data ################
    # Load ASR dataset
    print('Loading data...')
    train_dataset, dev_dataset, test_dataset = load_data(args)

    ################ Predicting ##################
    predictor = Predictor(args, test_dataset, features=opt.FEATURE_public)
    print('Number of utterances in data to test: ', test_dataset.num_utterances)
    print('Number of total data to test: ', len(test_dataset.data))
    predictor.visualize(train_dataset.data_for_use)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--type', default='train', dest='type', help='train | predict | visualize')
    parser.add_argument('-m', '--model', default='lambdamart', dest='model', help='fixed_weight | reg | lambdamart | lambdamart_tune')
    parser.add_argument('-v', '--vis', default='plot_tree', dest='vis', help='plot_tree | plot_feature_importance | plot_print_feature_shap | wer_plot')
    parser.add_argument('-f', '--file', default='test', dest='file', help='train | dev | test')
    parser.add_argument('-feat', '--feature', default='feature_public', dest='feature_to_use', help='enter the feature number')
    parser.add_argument('-p', '--postfix', default='', dest='postfix', help='postfix for the trained model to test')
    parser.add_argument('-i', '--iterations', default=2000, dest='iterations', help='iterations of training')
    parser.add_argument('-e', '--embed', default='bert', dest='embed', help='bert')
    parser.add_argument('-c', '--checkpoint_path', default='~/checkpoints/', dest='checkpoint_path', help='path for pretrained model (for local file only)')
    parser.add_argument('--preload', dest='preload', action='store_true')
    parser.add_argument('--no-preload', dest='preload', action='store_false')
    parser.set_defaults(preload=True)
    parser.add_argument('--cuda', type=int, help='specify the gpu id if needed, such as 0 or 1.', default=None)
    parser.add_argument('--dir_json', type=str, help='the path of json files specifying the evaluation details.')
    args = parser.parse_args()

    assert args.type in ['train', 'predict', 'visualize']
    assert args.model in ['fixed_weight', 'reg', 'lambdamart', 'lambdamart_tune']
    assert args.vis in ['plot_tree', 'plot_feature_importance', 'plot_print_feature_shap', 'wer_plot']
    assert args.file in ['train', 'dev', 'test']
    assert 'feature' in args.feature_to_use

    if not os.path.exists('./checkpoints/'):
        os.mkdir('./checkpoints/')

    if args.type == 'train':
        train(args)
    elif args.type == 'predict':
        predict(args)
    elif args.type == 'visualize':
        visualize(args)