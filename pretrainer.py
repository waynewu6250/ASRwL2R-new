import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, RMSprop
from transformers import BertTokenizer, AutoModel, BertConfig, AdamW, BertForPreTraining, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from datasets import Dataset

import pickle
import copy
import numpy as np
from tqdm import tqdm
import os
import re
import csv
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# from transformers import BertTokenizer
import math
from collections import defaultdict

from data.unstructured_dataset import UnStructuredDataset
from bert_data import CoreDataset, get_dataloader
from model.bert_model import BertEmbedding
from model.transformer import TransformerRescorer
from model.bert_rescorer import BertRescorer
from config import opt
from utils import get_groups, tokenize
# from ATMexperiments.wael.tokenizer import ATMTokenizerFast


class Pretrainer:

    def __init__(self, test_path, args, train_path=None, preload=True):
        """Main pretraining pipeline"""

        # data
        self.test_path = './data/espnet_parsed/'
        self.prefix = test_path.split('/')[3].split('.')[0]
        if train_path:
            self.train_dataset = UnStructuredDataset(train_path, opt=opt, preload=preload)
        self.test_dataset = UnStructuredDataset(test_path, opt=opt, preload=preload)

        # args
        self.pretrain_type = args.type
        self.pretrain_embed = args.embed
        self.checkpoint_path = args.checkpoint_path

        # tokenizer
        self.maxlen = 50
        self.REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
        self.BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # gpu
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch.backends.cudnn.enabled = False

        # bert model
        self.model = BertEmbedding(self.checkpoint_path)
        self.model = self.model.to(self.device)
        if os.path.exists('./checkpoints/model_{}.pth'.format(self.pretrain_embed)):
            self.model.load_state_dict(
                torch.load('./checkpoints/model_{}.pth'.format(self.pretrain_embed), map_location='cpu'))
            print('Pretrained bert model has been loaded...')


    #############################################################################################
    ################### Pipeline for generate similarity score (one-by-one) #####################
    #############################################################################################

    def generate_cosine_similarity(self):
        """Generate cosine similarity scores from glove and sbert pretrained embeddings"""

        embedding_matrix = None
        if self.pretrain_embed == 'glove':

            model = pd.read_csv('~/.vector_cache/glove.6B.300d.txt', sep=" ", index_col=0, header=None,
                                quoting=csv.QUOTE_NONE)
            vocab = {word: i for i, word in enumerate(model.index)}
            vocab['[UNK]'] = len(vocab)
            embedding_matrix = np.concatenate([model.to_numpy(), np.zeros((1, 300))])

            self.test_dataset.data['hyp_ids'] = self.test_dataset.data['hyp'].map(
                lambda x: tokenize(x, self.test_dataset, vocab, True))
            self.test_dataset.data['rewrite_ids'] = self.test_dataset.data['rewrite'].map(
                lambda x: tokenize(x, self.test_dataset, vocab, False))

        elif self.pretrain_embed == 'sbert':
            model = SentenceTransformer('paraphrase-distilroberta-base-v1')  # bert-base-nli-mean-tokens
            model = model.to(self.device)

        elif self.pretrain_embed == 'bert':
            model = self.model

        scores = self.generate_cs_scores(model, embedding_matrix)
        with open(self.test_path+'_rewrite_{}_pretrain.pkl'.format(self.pretrain_embed), 'wb') as f:
            pickle.dump(scores, f)


    def generate_cs_scores(self, model, embedding_matrix=None):
        """Generate text embeddings from glove and sBERT model and calculate cosine similarity scores"""

        final_scores = {}

        query_groups = get_groups(self.test_dataset.data.utt_id)
        for qid, a, b in tqdm(query_groups,
                              total=self.test_dataset.dataset.num_utterances,
                              desc='Batches',
                              unit=' batches',
                              ncols=80,
                              ):

            if self.pretrain_embed == 'glove':

                rewrite = [self.test_dataset.data.loc[a]['rewrite_ids']]
                hyps = [hyp for hyp in self.test_dataset.data.loc[a:b - 1]['hyp_ids'].values]
                sentences = rewrite + hyps

                sentence_embeddings = []
                for sent in sentences:
                    sentence_embeddings.append(embedding_matrix[sent].mean(axis=0))

            elif self.pretrain_embed == 'sbert':
                rewrite = [self.test_dataset.data.loc[a]['rewrite']]
                hyps = [self.test_dataset.remove_wake_words(hyp) for hyp in self.test_dataset.data.loc[a:b - 1]['hyp'].values]

                sentences = [rewrite] + hyps

                sentence_embeddings = model.encode(sentences)

            elif self.pretrain_embed == 'bert':
                rewrite = self.test_dataset.data.loc[a]['rewrite']
                # Uncomment the following line to use ground truth as rewrite
                # rewrite = [self.test_dataset.data.loc[a]['truth'] if self.test_dataset.data.loc[a]['truth'] != '' else self.test_dataset.data.loc[a]['rewrite']]
                hyps = [self.test_dataset.remove_wake_words(str(hyp)) for hyp in
                        self.test_dataset.data.loc[a:b - 1]['hyp'].values]

                sentences = [rewrite] + hyps

                tokenized_data = [self.text_prepare(text) for text in sentences]

                input_ids = tf.keras.preprocessing.sequence.pad_sequences(tokenized_data, maxlen=self.maxlen,
                                                                          dtype="long",
                                                                          truncating="post", padding="post")
                attention_masks = []
                for seq in input_ids:
                    seq_mask = [float(i > 0) for i in seq]
                    attention_masks.append(seq_mask)

                captions_t = torch.LongTensor(input_ids).to(self.device)
                mask = torch.LongTensor(attention_masks).to(self.device)
                with torch.no_grad():
                    outputs = model(captions_t, mask)
                pooled_output = outputs.hidden_states[-1][:, 0, :]  # (b,h)
                sentence_embeddings = pooled_output.data.cpu().numpy()

            final_scores[qid] = cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])[0]

        return final_scores

    def text_prepare(self, text):
        """ prepare text data
            :param text: a string
            :return text: a modified string
        """

        text = text.lower()  # lowercase text
        text = re.sub(self.REPLACE_BY_SPACE_RE, ' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = re.sub(self.BAD_SYMBOLS_RE, '', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
        text = re.sub(r"[ ]+", " ", text)
        text = re.sub(r"\!+", "!", text)
        text = re.sub(r"\,+", ",", text)
        text = re.sub(r"\?+", "?", text)

        text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(text)
        tokenized_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        text = tokenized_ids

        return text



    #############################################################################################
    ################ Pipeline for pretraining/generating embeddings (batch-wise) ################
    #############################################################################################

    def pretrain_bert(self):
        """Pretrain the BERT model for train/dev/text datasets"""

        def tokenize_function(examples):
            """Helper function for tokenizing the text"""
            return self.tokenizer(examples["text"])

        def group_texts(examples):
            """Helper function for grouping the text"""
            block_size = 128

            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])

            total_length = (total_length // block_size) * block_size

            # Split by chunks of max_len.
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Uncomment the following line for large pretraining
        # with open('/home/ec2-user/large_pretrain_text.pkl', 'rb') as f:
        #     texts = pickle.load(f)
        # print('Number of texts pretrained: ', len(texts))

        df_train = {'text': self.train_dataset.raw_texts}
        dataset_train = Dataset.from_dict(df_train)
        tokenized_datasets = dataset_train.map(tokenize_function, batched=True, num_proc=1, remove_columns=["text"])
        train_d = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=1000,
            num_proc=1,
        )

        df_test = {'text': self.test_dataset.raw_texts}
        dataset_test = Dataset.from_dict(df_test)
        tokenized_datasets = dataset_test.map(tokenize_function, batched=True, num_proc=1, remove_columns=["text"])
        test_d = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=1000,
            num_proc=1,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)

        training_args = TrainingArguments(
            output_dir="/home/ec2-user/pytorch_model_{}".format(self.pretrain_embed),
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            num_train_epochs=20,
            eval_steps = 10,  # Evaluation and Save happens every 10 steps
            save_total_limit = 5,  # Only last 5 models are saved. Older ones are deleted.
            load_best_model_at_end = True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_d,
            eval_dataset=test_d,
            data_collator=data_collator,
        )
        trainer.train()

        eval_results = trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        torch.save(trainer.model.state_dict(), '/home/ec2-user/model_{}.pth'.format(self.pretrain_embed))

    def generate_bert_scores_or_embeddings(self, type):
        """Generate text embeddings/scores from BERT model for train/dev/text datasets"""

        dataloader = get_dataloader(self.test_dataset, utterance_mode='single', pretrain_embed=self.pretrain_embed)

        model = self.model

        embeddings = []
        losses = []

        for (captions_t, masks) in tqdm(dataloader, total=len(dataloader),
                                        desc='Batches',
                                        unit=' batches',
                                        ncols=80,
                                        ):
            b, t = captions_t.shape
            captions_t = captions_t.to(self.device)
            masks = masks.to(self.device)
            with torch.no_grad():

                if type == 'embed_score':
                    # scores
                    outputs = model(captions_t, masks) 
                    for ii in range(len(captions_t)):
                        loss = nn.CrossEntropyLoss()(outputs.logits[ii], captions_t[ii])
                        losses.append(loss.item())
                elif type == 'embed_raw':
                    # embeddings
                    outputs = model(captions_t, masks)
                    pooled_output = outputs.hidden_states[-1][:, 0, :]  # (b,h)
                    for ii in range(len(captions_t)):
                        embedding = pooled_output[ii].data.cpu().numpy().reshape(-1)
                        embeddings.append(embedding)

        if type == 'embed_score':
            # Reshape the losses into group of nbest list for each utterance
            final_losses = defaultdict(list)
            query_groups = get_groups(self.test_dataset.data.data_id)
            counter = 0
            for qid, a, b in query_groups:
                for i in range(b - a):
                    final_losses[qid].append(losses[counter])
                    counter += 1

            # scores
            with open(self.test_path+self.prefix+'_{}_score_pretrain.pkl'.format(self.pretrain_embed), 'wb') as f:
                pickle.dump(final_losses, f)
            print(len(final_losses))

        elif type == 'embed_raw':
            # embeddings
            embeddings = np.vstack(embeddings)
            print(embeddings.shape)

            with open(self.test_path + '_embeddings.pkl', 'wb') as f:
                pickle.dump(embeddings, f)


    def pretrain_rescorer(self, rescorer_type, loss_type=None):
        """Pretrain the BERT/transformer rescorer for train/dev/text datasets"""

        train_dataloader = get_dataloader(self.train_dataset, utterance_mode='group', pretrain_embed=self.pretrain_embed)
        val_dataloader = get_dataloader(self.test_dataset, utterance_mode='group', pretrain_embed=self.pretrain_embed)

        if rescorer_type == 'transformer':
            model = TransformerRescorer(self.device, self.pretrain_embed, self.checkpoint_path, self.tokenizer.vocab_size, opt, pretrain_model=self.model)
            model_path = '/home/ec2-user/transformer_rescorer_{}.pth'.format(self.pretrain_embed)
        elif rescorer_type == 'bert':
            model = BertRescorer(self.device, self.pretrain_embed, self.checkpoint_path, self.tokenizer.vocab_size, opt, pretrain_model=self.model, loss_type=loss_type)
            model_path = './checkpoints/bert_rescorer_{}_{}_listwise.pth'.format(self.pretrain_embed, loss_type)
        model = model.to(self.device)

        # if os.path.exists(model_path):
        #     model.load_state_dict(torch.load(model_path))
        #     print('Pretrained transformer/bert model has been loaded...')

        optimizer = AdamW(model.parameters(), weight_decay=0.01, lr=opt.learning_rate_bert)

        best_accuracy = 0
        best_loss = 300

        # Start training
        for epoch in range(opt.epochs):
            print("====== epoch %d / %d: ======" % (epoch + 1, opt.epochs))

            # Training Phase
            total_train_loss = 0
            total_train_hyps = 0
            total_train_corrects = 0
            model.train()
            for i, batch in enumerate(tqdm(train_dataloader,
                                           total=len(train_dataloader),
                                           desc='Batches',
                                           unit=' batches',
                                           ncols=80)):

                optimizer.zero_grad()
                hidden = model(batch)
                if rescorer_type == 'transformer':
                    train_loss = model.ce_and_mwer_loss(hidden, batch)
                    # Uncomment the following line to switch loss
                    # train_loss = model.cross_entropy_loss(hidden, batch)
                    # train_loss = model.mwer_loss(hidden, batch)
                else:
                    train_loss, batch_hyps, batch_corrects = model.confidence_loss(hidden, batch)

                train_loss.backward()
                optimizer.step()

                total_train_loss += train_loss.detach().cpu().item()
                if rescorer_type != 'transformer':
                    total_train_hyps += batch_hyps.detach().cpu().item()
                    total_train_corrects += batch_corrects.detach().cpu().item()

            if rescorer_type == 'transformer':
                print('Average train loss: {:.4f} '.format(total_train_loss))
            else:
                average_train_loss = total_train_loss / total_train_hyps
                average_train_accuracy = total_train_corrects / total_train_hyps
                print('Average train loss: {:.4f} '.format(average_train_loss))
                print('Average train accuracy: {:.4f} '.format(average_train_accuracy))

            # Validation Phase
            total_val_loss = 0
            total_val_hyps = 0
            total_val_corrects = 0
            model.eval()
            for i, batch in enumerate(tqdm(val_dataloader,
                                           total=len(val_dataloader),
                                           desc='Batches',
                                           unit=' batches',
                                           ncols=80)):
                with torch.no_grad():
                    hidden = model(batch)
                    if rescorer_type == 'transformer':
                        val_loss = model.ce_and_mwer_loss(hidden, batch)
                        # Uncomment the following line to switch loss
                        # val_loss = model.cross_entropy_loss(hidden, batch)
                        # val_loss = model.mwer_loss(hidden, batch)
                    else:
                        val_loss, batch_hyps, batch_corrects = model.confidence_loss(hidden, batch)

                total_val_loss += val_loss.detach().cpu().item()
                if rescorer_type != 'transformer':
                    total_val_hyps += batch_hyps.detach().cpu().item()
                    total_val_corrects += batch_corrects.detach().cpu().item()

            if rescorer_type == 'transformer':
                print('Average val loss: {:.4f} '.format(total_val_loss))

                if total_val_loss < best_loss:
                    print('saving with loss of {}'.format(total_val_loss),
                          'improved over previous {}'.format(best_loss))
                    best_loss = total_val_loss
                    torch.save(model.state_dict(), model_path)
            else:
                average_val_loss = total_val_loss / total_val_hyps
                average_val_accuracy = total_val_corrects / total_val_hyps
                print('Average val loss: {:.4f} '.format(average_val_loss))
                print('Average val accuracy: {:.4f} '.format(average_val_accuracy))

                ## Uncomment the following line to save based on best loss
                # if average_val_loss < best_loss:
                #     print('saving with loss of {}'.format(average_val_loss),
                #           'improved over previous {}'.format(best_loss))
                #     best_loss = average_val_loss
                #     torch.save(model.state_dict(),'/home/ec2-user/bert_rescorer_{}_{}_listwise_rewrite.pth'.format(self.pretrain_embed,loss_type))

                ## Save based on best accuracy
                if average_val_accuracy > best_accuracy:
                    print('saving with accuracy of {}'.format(average_val_accuracy),
                          'improved over previous {}'.format(best_accuracy))
                    best_accuracy = average_val_accuracy
                    torch.save(model.state_dict(), './checkpoints/bert_rescorer_{}_{}_listwise.pth'.format(self.pretrain_embed, loss_type))

        print()
        print('Best total val loss: {:.4f}'.format(best_loss))

    def predict_rescorer(self, rescorer_type, loss_type=None):
        """Predict scores based on the BERT/transformer rescorer for train/dev/text datasets"""

        dataloader = get_dataloader(self.test_dataset, utterance_mode='group', pretrain_embed=self.pretrain_embed)

        if rescorer_type == 'transformer':
            model = TransformerRescorer(self.device, self.pretrain_embed, self.checkpoint_path,
                                        self.tokenizer.vocab_size, opt, pretrain_model=self.model)
            model_path = '/home/ec2-user/transformer_rescorer_{}.pth'.format(self.pretrain_embed)
        elif rescorer_type == 'bert':
            model = BertRescorer(self.device, self.pretrain_embed, self.checkpoint_path, self.tokenizer.vocab_size, opt, pretrain_model=self.model, loss_type=loss_type)
            model_path = './checkpoints/bert_rescorer_{}_{}_listwise.pth'.format(self.pretrain_embed, loss_type)
        model = model.to(self.device)
        model.load_state_dict(torch.load(model_path))
        print('Pretrained transformer/bert model has been loaded...')

        scores = defaultdict(list)
        counter = 0
        model.eval()
        for i, batch in enumerate(tqdm(dataloader,
                                       total=len(dataloader),
                                       desc='Batches',
                                       unit=' batches',
                                       ncols=80)):

            (result_ids, result_token_masks, result_masks, lengths, result_audio, result_wers, truth_ids,
             truth_token_masks, result_asr_features) = batch
            result_ids = result_ids.to(self.device)  # (b, d, t)
            lengths = lengths.to(self.device)  # (b,)

            b, d, t = result_ids.shape
            targets = Variable(torch.zeros(b, d, t)).to(self.device)
            targets[:, :, :-1] = result_ids[:, :, 1:].to(self.device)
            with torch.no_grad():
                logits = model(batch)
            if rescorer_type == 'transformer':
                logits = logits.view(b, d, t, -1)
                for ii in range(b):
                    for j in range(lengths[ii]):
                        loss = nn.CrossEntropyLoss()(logits[ii][j].reshape(t, -1), targets[ii][j].long())
                        scores[self.test_dataset.data.utt_id[counter]].append(loss.item())
                        counter += 1

            elif rescorer_type == 'bert':
                logits = logits.view(b, d, 1) if loss_type != 'margin' else logits[0].view(b, d, 1)
                for ii in range(b):
                    if loss_type == 'ce':
                        scores_pred = logits[ii][:lengths[ii]].unsqueeze(0)
                        scores_prob = nn.Softmax(dim=1)(scores_pred).squeeze(0)
                        for j in range(lengths[ii]):
                            scores[self.test_dataset.data.utt_id[counter]].append(scores_prob[j].item())
                            counter += 1
                    else:
                        for j in range(lengths[ii]):
                            if loss_type == 'regression' or loss_type == 'kl_div':
                                score = logits[ii][j].item()
                            else:
                                score = nn.Sigmoid()(logits[ii][j]).item()
                            scores[self.test_dataset.data.utt_id[counter]].append(score)
                            counter += 1

        print('Total utterances: ', len(scores))

        with open(self.test_path+self.prefix+'_{}_{}_rescorer_{}_listwise.pkl'.format(self.pretrain_embed, rescorer_type, loss_type), 'wb') as f:
            pickle.dump(scores, f)



























