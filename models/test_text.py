from train_text import extract_fast_text_features
import argparse
import pickle as cPickle
import re
import pandas as pd


def predict_text_labels(data, fasttext_pretrained_model, svm_model,
                        classes_file):
    """
    segment-level classification of text to classify into aggregated classes
    :param data: string(text) that we want to classify
    :param fasttext_pretrained_model: fasttext model or path to fasttextmodel
    :param svm_model: path to svm model
    :return: a dictionary that has as elements the classes and as values
    the percentage (%) that this data belongs to each class
    """
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', data)
    num_of_samples = len(sentences)
    feature_matrix = extract_fast_text_features(sentences,
                                                fasttext_pretrained_model)
    print(feature_matrix.shape)
    with open(svm_model, 'rb') as fid:
        classifier = cPickle.load(fid)
        mean = cPickle.load(fid)
        std = cPickle.load(fid)
    df = pd.read_csv(classes_file)
    classes = df['classes'].tolist()
    dictionary = {}
    for label in classes:
        dictionary[label] = 0
    for sample in feature_matrix: # for each sentence:
        # predict + normalization
        class_id = classifier.predict((sample - mean) / std)
        label = class_id[0]
        dictionary[label] += 1
    for label in dictionary:
        # TODO: Replace this aggregation by posterior-based aggregation
        dictionary[label] = (dictionary[label] * 100) / num_of_samples
    return dictionary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="the path of our data samples "
                             "(csv with column transcriptions "
                             "and one column labels)")
    parser.add_argument("-p", "--pretrained",
                        help="the path of fasttext pretrained model "
                             "(.bin file)")
    parser.add_argument("-c", "--classifier",
                        help="the path of the classifier")
    parser.add_argument("-n", "--names",
                        help="the path of csv file that contains the"
                              " name of classes of this specific model")
    args = parser.parse_args()
    predict_text_labels(args.input, args.pretrained,
                        args.classifier, args.names)

























