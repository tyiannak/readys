from models.train_text import extract_fast_text_features
from models.train_text import load_text_embeddings
import argparse
import pickle as cPickle
import pandas as pd


def extract_features(data, fasttext_pretrained_model, embeddings_limit=None):
    """
    Extract features from text segments
    :param data: list of samples (text-segments)
    :param fasttext_pretrained_model:  the fast text pretrained model loaded
    :param embeddings_limit: embeddings_limit: limit of the number of embeddings
        If None, then the whole set of embeddings is loaded.
    :return:
    -feature_matrix: features of all samples (n samples x m features)
    -num_of_samples: number of samples
    """
    num_of_samples = len(data)
    feature_matrix = extract_fast_text_features(
        data, fasttext_pretrained_model,
        embeddings_limit)
    return feature_matrix , num_of_samples


def predict_text_labels(feature_matrix, num_of_samples, svm_model,
                        classes_file):
    """
    segment-level classification of text to classify into aggregated classes
    :param feature_matrix: features of all samples (n samples x m features)
    :param num_of_samples: number of samples
    :param svm_model: path to svm model
    :param classes_file: the csv file with classes names
    :return:
    -dictionary : a dictionary that has as elements the classes and as values
    the percentage (%) that this data belongs to each class
    -predicted_labels : a list of predicted labels of all samples
    """

    with open(svm_model, 'rb') as fid:
        classifier = cPickle.load(fid)
        mean = cPickle.load(fid)
        std = cPickle.load(fid)
    df = pd.read_csv(classes_file)
    classes = df['classes'].tolist()
    dictionary = {}
    predicted_labels = []
    for label in classes:
        dictionary[label] = 0
    for sample in feature_matrix: # for each sentence:
        # predict
        class_id = classifier.predict((sample - mean) / std)
        label = class_id[0]
        predicted_labels.append(label)
        dictionary[label] += 1
    for label in dictionary:
        # TODO: Replace this aggregation by posterior-based aggregation
        dictionary[label] = (dictionary[label] * 100) / num_of_samples
    return dictionary , predicted_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="a string containing sentence(s)")
    parser.add_argument("-p", "--pretrained",
                        help="the path of fasttext pretrained model "
                             "(.bin file)")
    parser.add_argument("-c", "--classifier",
                        help="the path of the classifier")
    parser.add_argument("-n", "--names",
                        help="the path of csv file that contains the"
                              " name of classes of this specific model")
    parser.add_argument('-l', '--embeddings_limit', required=False,
                        default=None, type=int,
                        help='Strategy to apply in transfer learning: 0 or 1.')

    args = parser.parse_args()

    text_embed_model = load_text_embeddings(args.pretrained,
                                            args.embeddings_limit)
    feature_matrix , num_of_samples = extract_features(args.input,
                                                       text_embed_model,
                                                       args.embeddings_limit)
    results = predict_text_labels(feature_matrix,num_of_samples,
                                  args.classifier, args.names)
    print(results)

























