from feature_extraction import TextFeatureExtraction
import argparse
import fasttext
import pickle
from nltk.tokenize import sent_tokenize


def extract_features(data, feature_extractor):
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
    data = sent_tokenize(data)
    num_of_samples = len(data)
    feature_matrix = feature_extractor.transform(data)
    if feature_matrix.shape[0] == 1:
        feature_matrix.reshape(1, -1)
    return feature_matrix, num_of_samples


def predict_text_labels(feature_matrix, num_of_samples, model):
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

    model_dict = pickle.load(open(model, 'rb'))
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


def fassttext_predict(data, model_dict):

    data = sent_tokenize(data)

    model_path = model_dict['fasttext_model']
    model = fasttext.load_model(model_path)

    return model.predict(data)


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
    model_dict = pickle.load(open(args.classifier, 'rb'))
    if model_dict['text_classifier']['fasttext']:
        results = fassttext_predict(args.input, model_dict)
        print(results)
    else:
        feature_extractor = TextFeatureExtraction(args.pretrained,
                                                  args.embeddings_limit)
        feature_matrix, num_of_samples = extract_features(args.input, feature_extractor)
        results = predict_text_labels(feature_matrix, num_of_samples,
                                      args.classifier)
    print(results)

























