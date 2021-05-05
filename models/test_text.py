"""
Use segment-level text classifiers to predict segment-level (sentence)
decisions AND recording-level aggregates
"""

import sys
import os
import argparse
import torch
import numpy as np
from nltk.tokenize import sent_tokenize
from feature_extraction import bert_embeddings

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../'))

from models.feature_extraction import TextFeatureExtraction
from models.utils import load_text_classifier_attributes


def basic_segment_classifier_predict(feature_matrix, classifier,classes):
    """
    segment-level classification of text to classify into aggregated classes.
    Used only for basic classifiers (i.e. svm or xgboost)
    :param feature_matrix: features of all samples (n samples x m features)
    :param classifier: the classifier loaded
    :param classes: the class names loaded from the classifier file

    :return: 1. dictionary : a dictionary that has as elements the classes
                and as values the percentage (%) that this data belongs to
                each class
             2. predicted_labels : a list of predicted labels of all samples
    """

    num_of_samples = feature_matrix.shape[0]
    dictionary = {}
    predicted_labels = []
    for label in classes:
        dictionary[label] = 0
    for sample in feature_matrix: # for each sentence:
        sample = sample.reshape(1, -1)
        class_id = classifier.predict(sample)
        label = class_id[0]
        predicted_labels.append(label)
        dictionary[label] += 1
    for label in dictionary:
        # TODO: Replace this aggregation by posterior-based aggregation
        dictionary[label] = (dictionary[label] * 100) / num_of_samples
    return dictionary, predicted_labels


def predict(pure_data, embeddings_type, classifier, classes, pretrained, embeddings_limit):
    """
    Checks the type of the classifier and decides how to predict labels
    on test data.
    :param data: list of string segments or string with sentences
    :param classifier: the classifier loaded
    :param pretrained: the pretrained model loaded
    :embeddings_limit: the embeddings limit
    :return: 1. dictionary : a dictionary that has as elements the classes
                and as values the percentage (%) that this data belongs to
                each class
             2. predicted_labels : a list of predicted labels of all samples
    """
    if isinstance(pure_data, str):
        data = sent_tokenize(pure_data)
    if pretrained == None:
        dictionary = {}
        predicted_labels = classifier.predict(data)
        num_of_samples = len(predicted_labels[0])
        for label in classes:
            dictionary[label] = 0
        for sample in predicted_labels[0]:
            label = sample[0]
            dictionary[label] += 1
        for label in dictionary:
            # TODO: Replace this aggregation by posterior-based aggregation
            dictionary[label] = (dictionary[label] * 100) / num_of_samples
    elif embeddings_type == "bert":
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        feature_matrix, _ = bert_embeddings(pure_data, [0], pretrained, device=device)
        feature_matrix = np.array(feature_matrix)
        dictionary, predicted_labels = \
            basic_segment_classifier_predict(feature_matrix,
                                             classifier, classes)

    else:
        feature_extractor = TextFeatureExtraction(pretrained,
                                                  embeddings_limit)
        feature_matrix = feature_extractor.transform(data)
        dictionary, predicted_labels = \
            basic_segment_classifier_predict(feature_matrix,
                                             classifier,classes)

    return dictionary, predicted_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="a string containing sentence(s)")
    parser.add_argument("-c", "--classifier",
                        help="the path of the classifier")
    args = parser.parse_args()
    embeddings_type, classifier, classes, pretrained_path, pretrained, embeddings_limit, _ = \
        load_text_classifier_attributes(args.classifier)
    dictionary, _ = predict(args.input, embeddings_type, classifier,
                            classes,
                            pretrained,
                            embeddings_limit)
    print(dictionary)
