import numpy as np
import os
import glob2 as glob
import fasttext
from gensim.models import KeyedVectors
from utils import text_preprocess
from utils import folders_mapping
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO


class TextFeatureExtraction(object):
    def __init__(self, word_model_path, embeddings_limit=None):
        """
            Initializes a FeatureExtraction object by loading the fasttext
            text representation model
            :param word_model_path: path to the fasttext .bin file
            :param embeddings_limit: limit of the number of embeddings.
                If None, then the whole set of embeddings is loaded.
        """
        self.embeddings_limit = embeddings_limit
        self.embedding_model = word_model_path
        print("--> Loading the text embeddings model")
        if embeddings_limit:
            self.word_model = KeyedVectors.load_word2vec_format(word_model_path,
                                                                limit=embeddings_limit)
        else:
            self.word_model = fasttext.load_model(word_model_path)

    def fit(self):  # comply with scikit-learn transformer requirement
        return self

    def transform(self, docs):  # comply with scikit-learn transformer requirement
        doc_word_vector = self.sentence_features_list(docs)
        return doc_word_vector

    def sentence_features(self, sentence):
        """
           Given a sentence (example) extract a feature vector
           based on fasttext pretrained model

           :param transcriptions: list of samples-sentences ,
           :param text_emb_model : path of fasttext pretrained
           model (.vec file)
           :return: fasttext_pretrained_model: numpy array (n x 300) -->
                                               n samples(sentences) x 300
                                               dimensions(features) normalized
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
                    features.append(feature)
                except:
                    continue
            else:
                feature = self.word_model[word]
                features.append(feature)

        # average the feature vectors for all the words in a sentence-sample
        X = np.array(features)
        mu = np.mean(X, axis=0)
        # save one vector(300 dimensional) for every sample

        return mu

    def sentence_features_list(self, docs):
        print("--> Extracting text features")
        return np.vstack([self.sentence_features(sent) for sent in docs])


class AudioFeatureExtraction(object):
    def __init__(self, basic_features_params):
        """
            Initializes a FeatureExtraction object by loading the fasttext
            text representation model
            :param word_model_path: path to the fasttext .bin file
            :param embeddings_limit: limit of the number of embeddings.
                If None, then the whole set of embeddings is loaded.
        """
        self.basic_features_params = basic_features_params

    def fit(self):  # comply with scikit-learn transformer requirement
        return self

    def transform(self, folder):  # comply with scikit-learn transformer requirement

        print("--> Extracting audio features")
        filenames = []
        labels = []

        folders = [x[0] for x in os.walk(folder)]

        if folders:
            for folder in folders:
                for f in glob.iglob(os.path.join(folder, '*.wav')):
                    filenames.append(f)
                    labels.append(folder)

            folder2idx, idx2folder = folders_mapping(folders=folders)
            labels = list(map(lambda x: folder2idx[x], labels))
            labels = np.asarray(labels)

        else:
            filenames = [input]
        # Match filenames with labels
        sequences_short_features, feature_names = \
            self.extract_segment_features(filenames)

        sequences_short_features_stats = []
        for sequence in sequences_short_features:
            mu = np.mean(sequence, axis=1)
            sequences_short_features_stats.append(mu)

        sequences_short_features_stats = np.asarray(sequences_short_features_stats)

        return sequences_short_features_stats, labels

    @staticmethod
    def read_files(self, filenames):
        """Read file using pyAudioAnalysis"""

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
