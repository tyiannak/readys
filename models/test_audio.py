import argparse
from pyAudioAnalysis.audioSegmentation import mid_term_file_classification


def predict_audio_labels(audio_file, svm_model):
    """
    Predict audio class
    :param audio_file: path of wav file
    :param svm_model: path of saved svm classifier
    :return: a dictionary that has as elements the classes
        and as values the percentage (%) that this wav
        belongs to each class
    """
    labels, class_names, _, _ = mid_term_file_classification(
        audio_file, svm_model, "svm_rbf")
    num_of_samples = len(labels)
    dictionary = {}
    for label in class_names:
        dictionary[label] = 0
    for i in labels:
        index = int(i)
        key = class_names[index]
        dictionary[key] += 1
    for label in dictionary:
        dictionary[label] = (dictionary[label] * 100) / num_of_samples
    return dictionary


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
