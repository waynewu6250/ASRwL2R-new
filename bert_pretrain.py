from config import opt
from pretrainer import Pretrainer

def run(args):
    """Run pretraining before the main training/evaluation pipeline"""
    if args.type.find('pretrain') != -1:

        pretrainer = Pretrainer(opt.test_path, args, train_path=opt.train_path, preload=True)
        if args.type == 'pretrain_bert':
            pretrainer.pretrain_bert()
        elif args.type == 'pretrain_rescorer':
            pretrainer.pretrain_rescorer(rescorer_type=args.rescorer_type, loss_type=args.loss_type)

    else:
        paths = [opt.train_path, opt.test_path]
        for path in paths:
            pretrainer = Pretrainer(path, args)
            if args.type.find('embed') != -1:
                pretrainer.generate_bert_scores_or_embeddings(type=args.type)
            elif args.type == 'score':
                pretrainer.generate_cosine_similarity()
            elif args.type == 'predict_rescorer':
                pretrainer.predict_rescorer(rescorer_type=args.rescorer_type, loss_type=args.loss_type)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--type', default='score', dest='type', help='score | embed_raw | embed_score | pretrain_bert | pretrain_rescorer | predict_rescorer')
    parser.add_argument('-e', '--embed', default='glove', dest='embed', help='glove | sbert | bert | mulan')
    parser.add_argument('-c', '--checkpoint_path', default='./pytorch_model/', dest='checkpoint_path', help='path for pretrained model')
    parser.add_argument('-r', '--rescorer_type', default='bert', dest='rescorer_type', help='transformer | bert')
    parser.add_argument('-l', '--loss', default='regression', dest='loss_type', help='regression | bce | bce_mwer')
    args = parser.parse_args()
    assert args.type in ['score', 'embed_raw', 'embed_score', 'pretrain_bert', 'pretrain_rescorer', 'predict_rescorer']
    assert args.embed in ['glove', 'sbert', 'bert', 'mulan']
    assert args.rescorer_type in ['transformer', 'bert']

    run(args)



