import kaldi_io
import os
import json
from data.dataset_public import L2RDataset
from collections import defaultdict
import numpy as np

class UnStructuredDataset:
    """Dataset object for unstructured data like audio, texts, rewrites to preprocess before the main pipeline

    :parameter:
    dataset: L2RDataset object
    raw_texts: hypothesis raw texts
    audio_dic: dictionary of {utt_id: embeddings}
    """

    def __init__(self, file_path, opt=None, preload=True):
        """Initialize the object
        :param: file_path: train/dev/test file path prefix
        :param: opt: configurations defined in config.py
        :param: preload: whether to use the processed data or preprocess again
        """
        self.opt = opt
        self.preload = preload
        self.dataset = self.load_dataset(file_path)
        self.raw_texts = self.dataset.data['hyp'].to_numpy()
        # Uncomment the following line to use truth sentence instead of hypothesis
        # self.raw_texts = np.array(set(self.dataset.data['truth'].to_numpy()))

        self.data = self.dataset.data
        self.data_for_use = self.dataset.data_for_use
        self.audio_dic = self.dataset.audio_dic


    def load_dataset(self, path):
        """Load dataset object"""
        # paths = {path: prefix + file for path, file in zip(self.opt.paths, self.opt.files_to_use)}
        dataset = L2RDataset(path, None, features=self.opt.FEATURE_to_train, opt=self.opt, preload=self.preload)
        print('Number of utterances loaded:', dataset.num_utterances)
        print('------------------------------')
        return dataset

    @staticmethod
    def remove_wake_words(string):
        """Remove wake words for a given string"""
        wake_words = ['alexa', 'amazon', 'echo', 'computer']
        querywords = string.split()
        resultwords = [word for word in querywords if word.lower() not in wake_words]
        result = ' '.join(resultwords)

        return result
