from feature_extraction import TextFeatureExtraction
import argparse
import fasttext
import pickle
from nltk.tokenize import sent_tokenize


def basic_segment_classifier_predict(feature_matrix, model_dict):
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

    num_of_samples = feature_matrix.shape[0]
    classifier = model_dict['classifier']
    classes = model_dict['classifier_classnames']
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


def predict(data, classifier_path, pretrained):

    model_dict = pickle.load(open(classifier_path, 'rb'))
    if isinstance(data, str):
        data = sent_tokenize(data)
    if model_dict['classifier_type'] == 'fasttext':
        model_path = model_dict['fasttext_model']
        model = fasttext.load_model(model_path)

        preds = model.predict(data)
    else:
        embeddings_limit = model_dict['embeddings_limit']
        feature_extractor = TextFeatureExtraction(pretrained,
                                                  embeddings_limit)
        feature_matrix = feature_extractor.transform(data)
        preds = basic_segment_classifier_predict(feature_matrix, model_dict)

    return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="a string containing sentence(s)")
    parser.add_argument("-p", "--pretrained",
                        help="the path of fasttext pretrained model "
                             "(.bin file)")
    parser.add_argument("-c", "--classifier",
                        help="the path of the classifier")
    parser.add_argument('-l', '--embeddings_limit', required=False,
                        default=None, type=int,
                        help='Strategy to apply in transfer learning: 0 or 1.')

    args = parser.parse_args()
    preds = predict(args.input, args.classifier, args.pretrained)
    print(preds)











