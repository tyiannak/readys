"""
Contains all wrappers for text and audio segment-levevl
feature extraction
"""

import numpy as np
import glob2 as glob
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO
import os
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import BertModel
from collections import OrderedDict


sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../'))

from models.utils import text_preprocess
from models.utils import folders_mapping
from models.utils import bert_preprocessing, seed_torch


class SSTDataset(Dataset):

    def __init__(self, df, maxlen):

        self.df = df
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        sentence = self.df.loc[index, 'sentence']
        label = self.df.loc[index, 'label']

        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(
                self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]']

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()
        return tokens_ids_tensor, attn_mask, label


def bert_embeddings(sentences, labels, bert, device="cpu", inference=False, force_len=-1):
    """
    Extract embeddings using BERT
    :param sentences: list of sentences
    :param labels: list of corresponding sentence labels
    :param device: device to run
    :return: - embeddings: list of embeddings per sentence
             - labels: list of labels
    """

    print("--> Extracting Bert Embeddings")
    print("---> Running on: {}".format(device))

    torch.cuda.empty_cache()
    seed_torch()

    df, le, max_len = bert_preprocessing(sentences, labels)
    print("----> Class mapping: {}".format(le.classes_))

    if force_len < 0:
        dataset = SSTDataset(df, maxlen=max_len)
    else:
        dataset = SSTDataset(df, maxlen=force_len)
    if inference:
        batch_size = 1
    else:
        a = int(max_len / 32)
        batch_size = int(32 / pow(2, a))

    data_loader = DataLoader(dataset, batch_size=batch_size)

    bert.to(device)
    bert.eval()

    embeddings = []
    batch_labels = []
    with torch.no_grad():
        for seq, attn_masks, local_labels in data_loader:
            seq, attn_masks = seq.to(device), \
                                      attn_masks.to(device)
            outputs = bert(seq, attention_mask=attn_masks)
            hidden_states = outputs[2]
            batch_tokens = torch.stack(hidden_states[-4:]).sum(0)

            for token_vecs in batch_tokens:
                sentence_embedding = torch.mean(token_vecs, dim=0)
                embeddings.append(sentence_embedding.detach().cpu().numpy())

            batch_labels.append(local_labels.cpu().numpy())

    labels = [item for sublist in batch_labels for item in sublist]
    return embeddings, labels, max_len


class TextFeatureExtraction(object):
    def __init__(self, word_model, embeddings_limit=None):
        """
            Initializes a TextFeatureExtraction object
            :param word_model: the loaded fasttext model
            :param embeddings_limit: limit of the number of embeddings.
        """
        self.word_model = word_model
        self.embeddings_limit = embeddings_limit

    def fit(self):
        # comply with scikit-learn transformer requirement
        return self

    def transform(self, docs, labels):
        # comply with scikit-learn transformer requirement
        doc_word_vector, labels = self.sentence_features_list(docs, labels)
        return doc_word_vector, labels

    def sentence_features(self, sentence):
        """
           Given a segment (example) extracts a feature vector
           based on fasttext pretrained model

           :param sentence: a segment
           :return: mean feature vector of the segment's words
        """

        features = []
        sentence = text_preprocess(sentence)
        for word in sentence.split():  # for every word in the sentence
            # TODO: sum features instead of append to reduce complexity
            if self.embeddings_limit:
                try:
                    result = self.word_model.similar_by_word(word)
                    most_similar_key, similarity = result[0]
                    feature = self.word_model[most_similar_key]
                    if len(feature) == 300:
                        features.append(feature)
                except:
                    continue
            else:
                feature = self.word_model[word]
                features.append(feature)

        # average the feature vectors for all the words in a sentence-sample
        if features != []:
            X = np.array(features)
            mu = np.mean(X, axis=0)
        else:
            mu = []
        # save one vector(300 dimensional) for every sample

        return mu

    def sentence_features_list(self, docs, labels):
        """
        For each segment in docs, extracts the mean feature
        vector of the segment's words
        :param docs: list of segments
        :return: feature matrix for the whole dataset
        """
        print("--> Extracting text features")
        total_features = []
        new_labels = []
        for count, sent in enumerate(docs):
            sample_features = self.sentence_features(sent)
            if sample_features != []:
                total_features.append(sample_features)
                new_labels.append(labels[count])
        return np.vstack(total_features), new_labels


class AudioFeatureExtraction(object):
    def __init__(self, basic_features_params):
        """
            Initializes an AudioFeatureExtraction object by loading the
            basic feature extraction parameters
            :param basic_features_params: basic feature extraction parameters
        """
        self.basic_features_params = basic_features_params

    def fit(self):  # comply with scikit-learn transformer requirement
        return self

    def transform(self, folder):  # comply with scikit-learn transformer requirement
        """
        Extract features on a dataset which lies under the folder directory
        :param folder: the directory in which the dataset is located.
                        Each subfolder of the folder must contain instances
                        of a specific class.
        :return: 1. sequences_short_features_stats:  stats on segment
                        features for each dataset instance
                 2. labels: list of labels
                 3. idx2folder: a mapping from label numbers to label names
        """

        filenames = []
        labels = []

        folders = [x[0] for x in os.walk(folder)]
        folders = sorted(folders[1:])
        folder_names = [os.path.split(folder)[1] for folder in folders]

        if folders:
            for folder in folders:
                for f in glob.iglob(os.path.join(folder, '*.wav')):
                    filenames.append(f)
                    labels.append(os.path.split(folder)[1])
            folder2idx, idx2folder = folders_mapping(folders=folder_names)
            labels = list(map(lambda x: folder2idx[x], labels))
            labels = np.asarray(labels)

        else:
            filenames = [folder]
        print(labels)
        print(idx2folder)
        # Match filenames with labels
        sequences_short_features, feature_names = \
            self.extract_segment_features(filenames)

        sequences_short_features_stats = []
        for sequence in sequences_short_features:
            mu = np.mean(sequence, axis=1)
            sequences_short_features_stats.append(mu)

        sequences_short_features_stats = np.asarray(
            sequences_short_features_stats)

        return sequences_short_features_stats, labels, idx2folder

    @staticmethod
    def read_files(filenames):
        """Read audio file using pyAudioAnalysis"""

        # Consider same sampling frequencies
        sequences = []
        for file in filenames:
            fs, samples = aIO.read_audio_file(file)
            sequences.append(samples)

        sequences = np.asarray(sequences)

        return sequences, fs

    def extract_segment_features(self, filenames):
        """
        Extract segment features using pyAudioAnalysis

        Parameters
        ----------

        filenames :
            List of input audio filenames

        basic_features_params:
            Dictionary of parameters to consider.
            It must contain:
                - mid_window: window size for framing
                - mid_step: window step for framing
                - short_window: segment window size
                - short_step: segment window step

        Returns
        -------

        segment_features_all:
            List of stats on segment features
        feature_names:
            List of feature names

        """
        print("--> Extracting audio features")
        segment_features_all = []

        sequences, sampling_rate = self.read_files(filenames)

        mid_window = self.basic_features_params['mid_window']
        mid_step = self.basic_features_params['mid_step']
        short_window = self.basic_features_params['short_window']
        short_step = self.basic_features_params['short_step']

        for seq in sequences:
            (segment_features_stats, segment_features,
             feature_names) = aF.mid_feature_extraction(
                seq, sampling_rate, round(mid_window * sampling_rate),
                round(mid_step * sampling_rate),
                round(sampling_rate * short_window),
                round(sampling_rate * short_step))
            segment_features_stats = np.asarray(segment_features_stats)
            segment_features_all.append(segment_features_stats)

        return segment_features_all, feature_names
