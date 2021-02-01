import numpy as np
import fasttext
from gensim.models import KeyedVectors
from utils import text_preprocess


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
