"""
This file runs Masked Language Model. You provide a training file. Each line is interpreted as a sentence / paragraph.
Optionally, you can also provide a dev file.
The fine-tuned model is stored in the output/model_name folder.
Usage:
python pretrain_mlm.py model_name
"""
# "TODBERT/TOD-BERT-JNT-V1"
# "distilroberta-base"
# "roberta-base"

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
import sys
import gzip
import json
from datetime import datetime
import pandas as pd
from torch.utils.data import DataLoader

import numpy as np
from config import opt

model_name = sys.argv[1]
per_device_train_batch_size = 16

save_steps = 1000               #Save model every 1k steps
num_train_epochs = 3            #Number of epochs
use_fp16 = False                #Set to True, if your GPU supports FP16 operations
max_length = 100                #Max length for a text input
do_whole_word_mask = True       #If set to true, whole words are masked
mlm_prob = 0.15                 #Probability that a word is replaced by a [MASK] token

# Load the model
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

output_dir = "checkpoints/mlm-pretrained/"
print("Save checkpoints to:", output_dir)


##### Load training datasets

# Load data
data = pd.read_csv('./data/espnet_parsed/train-all-pl.csv')
sentences = data.truth.values
np.random.seed(42)
indices = np.random.permutation(len(sentences))
idd = int(len(sentences)*0.9)
train_sentences = np.array(sentences, dtype=object)[indices[:idd]]
dev_sentences = np.array(sentences, dtype=object)[indices[idd:]]


#A dataset wrapper, that tokenizes our data on-the-fly
class TokenizedSentencesDataset:
    def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item):
        # if not self.cache_tokenization:
        return self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)

        # if isinstance(self.sentences[item], str):
        #     self.sentences[item] = self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)
        # return self.sentences[item]

    def __len__(self):
        return len(self.sentences)

train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer, max_length)
dev_dataset = TokenizedSentencesDataset(dev_sentences, tokenizer, max_length, cache_tokenization=True) if len(dev_sentences) > 0 else None


##### Training arguments

if do_whole_word_mask:
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    evaluation_strategy="steps" if dev_dataset is not None else "no",
    per_device_train_batch_size=per_device_train_batch_size,
    eval_steps=save_steps,
    save_steps=save_steps,
    logging_steps=save_steps,
    save_total_limit=1,
    prediction_loss_only=True,
    fp16=use_fp16
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset
)

print("Save tokenizer to:", output_dir)
tokenizer.save_pretrained(output_dir)

trainer.train()

print("Save model to:", output_dir)
model.save_pretrained(output_dir)

print("Training done")