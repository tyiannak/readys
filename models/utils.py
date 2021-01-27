import re
import csv
import numpy as np
import pandas as pd
import fasttext
from gensim.models import KeyedVectors
import num2words
from collections import Counter
from sklearn.model_selection import train_test_split


def text_preprocess(document):
    """
        Basic text preprocessing
        :param document: string containing input text
        :return: updated text
    """

    document.rstrip("\n")

    # Remove contractions
    # document = contractions.fix(document)

    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Numbers to words
    after_spliting = document.split()

    for index, word in enumerate(after_spliting):
        if word.isdigit():
            after_spliting[index] = num2words(word)
    document = ' '.join(after_spliting)

    # Converting to Lowercase
    preprocessed_text = document.lower()

    return preprocessed_text


def load_dataset(data, class_file_name, hop_samples=None):
    # load our samples
    df = pd.read_csv(data)
    transcriptions = df['transcriptions'].tolist()
    docs = []
    for sentence in transcriptions:
        docs.append(text_preprocess(sentence))
    labels = df['labels']

    if hop_samples:
        docs = docs[::hop_samples]
        labels = labels[::hop_samples]
    a = np.unique(labels)
    df = pd.DataFrame(columns=['classes'])
    df['classes'] = a
    df.to_csv(class_file_name, index=False)
    labels = labels.tolist()

    return docs, labels


def convert_to_fasttext_data(labels, transcriptions, filename):
    data = [label + " " + trans for label, trans in zip(labels, transcriptions)]
    df = pd.DataFrame(data)
    df.to_csv('data.txt', index=False, sep=' ',
              header=None, quoting=csv.QUOTE_NONE,
              quotechar="", escapechar=" ")


def check_balance(labels):

    class_samples = Counter(labels)
    balanced_degrees = 1/len(class_samples)
    imbalanced_degrees = {}
    for label in class_samples:
        imbalanced_degrees[label] = class_samples[label]/len(labels)
    print('--> Dataset samples per class: \n    {}'.format(imbalanced_degrees))

    is_imbalanced = False
    strategy = {}
    for label in imbalanced_degrees:
        if imbalanced_degrees[label] < 0.8 * balanced_degrees:
            is_imbalanced = True
            print("--> The class {} is undersampled.".format(label))

    return is_imbalanced


def split_data(x, y, test_size=0.2, fasttext_data=False, seed=None):

    if fasttext_data:
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=test_size, random_state=seed,
                                                            stratify=y)

        convert_to_fasttext_data(y_train, x_train, 'train.txt')
        convert_to_fasttext_data(y_test, x_test, 'test.txt')
        print("Splitted dataset into train (saved on train.txt) and test (saved on test.txt) subsets.")
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=test_size, random_state=seed,
                                                            stratify=y)
        return x_train, x_test, y_train, y_test


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
        # mu = np.mean(X, axis=0) + eps
        # std = np.std(X, axis=0) + eps
        mu = np.mean(X, axis=0)
        # save one vector(300 dimensional) for every sample

        return mu

    def sentence_features_list(self, docs):
        print("--> Extracting text features")
        return np.vstack([self.sentence_features(sent) for sent in docs])


