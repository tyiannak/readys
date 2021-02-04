import argparse
import pickle5 as pickle
from collections import Counter
from feature_extraction import AudioFeatureExtraction


def predict_audio_labels(filename, classifier_path):
    """
    Extract audio segments from the audio file and predict
    the class of each segment.
    :param filename: path of wav file
    :param classifier_path: path to a dictionary containing:
        - classifier: the output classifier
        - classifier_type: the type of the output classifier (i.e basic)
        - class_mapping: a mapping from label numbers to label names
        - basic_features_params: parameters for the segment based
                                feature extraction

    :return: 1. dictionary : a dictionary that has as elements the classes
                and as values the percentage (%) that this data belongs to
                each class
             2. predicted_labels : a list of predicted labels of all samples
    """
    model_dict = pickle.load(open(classifier_path, 'rb'))

    # Feature extraction
    feature_extractor = AudioFeatureExtraction(
        model_dict['basic_features_params'])
    segment_features, _ = \
        feature_extractor.extract_segment_features([filename])
    segment_features = segment_features[0].transpose()

    # Predict
    classifier = model_dict['classifier']

    preds = []
    for segment in segment_features:
        segment = segment.reshape(1, -1)
        pred = classifier.predict(segment)
        preds.append(pred[0])

    # Stats on predictions
    num_of_samples = len(preds)
    class_mapping = model_dict['class_mapping']
    preds_classnames = list(map(lambda x: class_mapping[x], preds))

    class_count = Counter(preds_classnames)
    pred_dict = {}
    for i in class_mapping:
        j = class_mapping[i]
        pred_dict[j] = 0.0
    for label in class_count:
        pred_dict[label] = (class_count[label]/num_of_samples)*100

    return pred_dict, preds_classnames


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="the path of audio file" )
    parser.add_argument("-c", "--classifier", required=True,
                        help="the path of the classifier")
    args = parser.parse_args()
    results = predict_audio_labels(args.input, args.classifier)
    print("Predicted probability percentage "
          "for each class:\n     {}".format(results))
