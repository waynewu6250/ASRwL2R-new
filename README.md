# A LEARNING-TO-RANK APPROACH TO ASR RESCORING
ICASSP 2022 paper

Open source codes of the learning-to-rank based ASR rescoring on librispeech scores generated from ESPNet.

# 1. Quick set-up

Install the dependencies:
```
pip3 install -r requirements.txt
```

# 2. Folder structure

```
ASRwL2R-public/
   └── data/
       └── libri_subset/ (librispeech scores folder)
       ├── parse_public.py (parse librispeech score from data/libri_subset)
       ├── dataset_public.py  (dataset file)
       ├── unstructured_dataset.py (unstructured data like texts, audio for finetuning/NN model)
   └── model/
       ├── lambdamart.py (hyperparameter tuning)
       ├── bert_model.py (bert model)
       ├── bert_rescorer.py (ASR BERT confidence model (ASRCM))
       ├── bert_ltr_rescorer.py (End-to-end BERT listwise LTR model)
       ├── transformer.py (second pass transformer LM rescorer)
       ├── transformer_fuser.py (transformer structure for ASRCM)
   trainer.py (main training pipeline)
   predictor.py (main evaluation pipeline)
   config.py (configurations)
   utils.py
   main.py
   bert_pretrain.py (bert finetuning entry)
   bert_data.py (bert dataloader)
   bert_data_ltr.py (bert dataloader for BERT LTR model)
   requirements.txt
```


# 3. Instructions

## 1. Quick start

There are three steps to run this repo: 
1) parse the librispeech score files in `data/`
2) generate the features and put it in the training csv for LambdaMART training
3) train & predict LambdaMART LTR model

### STEP 1: Parse data
Parse the librispeech score files in `data/`
   ```
   python parse_public.py -t phase_1
   ```

### STEP 2: Generate features
1. Option 1: Directly use the sample extracted features: (stored in `data/libri_subset/`)
2. Option 2: Follow the feature guide below to generate features.
   Ex. 
   1) Train the model
   ```
   python bert_pretrain.py -t pretrain_rescorer -r bert -e bert -l bce
   ```
   2) Generate the scores
   ```
   python bert_pretrain.py -t predict_rescorer -r bert -e bert -l bce
   ```
   3) Put the score in csv again
   ```
   python parse_public.py -t phase_2
   ```

### STEP 3: Train & predict LTR model
1) Train LambdaMART LTR model
   
   (featureID is defined in config.py. Ex. For FEATURE21 (BEST), you enter `bash run.sh lambdamart feature21`)
   ```
   python main.py -t train -m lambdamart -feat feature_public
   ```
2) Evaluate LambdaMART LTR model
   ```
   python main.py -t predict -m lambdamart -feat feature_public
   ```

## 2. Feature guide

We can generate different sets of features by training upstream models and generating scores.

### STEP 1: Training

For `[checkpoint_path]`, bert uses `/home/ec2-user/pytorch_model/` and mulan uses `/home/ec2-user/pretrained/`
  
  * Fine-tune Bert model:
     ```
     python bert_pretrain.py -t pretrain_bert -e bert -c [checkpoint_path]
     ```

  * Train ASR BERT-based confidence model:
    
    (Available loss type: `bce`, `bce_mwer`, `ce`, `kl_div`, `regression`)
     ```
     python bert_pretrain.py -t pretrain_rescorer -r bert -e bert -c [checkpoint_path] -l [loss_type]
     ```

  * Train a Transformer rescorer:
     ```
     python bert_pretrain.py -t pretrain_rescorer -r transformer -e bert -c [checkpoint_path]
     ```
      
### STEP 2: Generating scores
  
  * Generate cosine similarity:
     ```
     python bert_pretrain.py -t score -e [glove/sbert/bert] -c [checkpoint_path]
     ```
  
  * Generate text embeddings:
     ```
     python bert_pretrain.py -t embed_raw -e bert -c [checkpoint_path]
     ```
  
  * Generate BERT MLM scores:
     ```
     python bert_pretrain.py -t embed_score -e bert -c [checkpoint_path]
     ```
  
  * Generate Transformer rescorer score:
     ```
     python bert_pretrain.py -t predict_rescorer -r transformer -e bert -c [checkpoint_path]
     ```

  * Generate ASR confidence model score:

    (Available loss type: `bce`, `bce_mwer`, `ce`, `kl_div`, `regression`)
     ```
     python bert_pretrain.py -t predict_rescorer -r bert -e bert -c [checkpoint_path] -l [loss_type]
     ```