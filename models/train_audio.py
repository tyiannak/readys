from pyAudioAnalysis.audioTrainTest import extract_features_and_train
import os
import argparse

# TODO add these in a config file
SEGMENT_LENGTH = 3
SEGMENT_STEP = 3
ST_WINDOW = 0.05
ST_STEP = 0.05
AUDIO_SEGMENT_CLASSIFIER_TYPE = "svm_rbf"


def train_svm(files_path, output_model):
    """
    Train audio models
    :param files_path: directory which contains
        audio organized in folders of classes
    :param output_model: path to save the output model
    :return: the name of the saved model
    """

    dirs = [x[0] for x in os.walk(files_path)]
    dirs = dirs[1:]
    extract_features_and_train(dirs,
                               SEGMENT_LENGTH, SEGMENT_STEP,
                               ST_WINDOW, ST_STEP,
                               AUDIO_SEGMENT_CLASSIFIER_TYPE,
                               output_model)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="the path of directory which "
                             "contains audio organized "
                             "in folders of classes")
    parser.add_argument("-o", "--outputmodelpath", required=True,
                        help="path to the final svm model to be saved")
    args = parser.parse_args()
    train_svm(args.input, args.outputmodelpath)
