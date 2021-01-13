from pyAudioAnalysis.audioTrainTest import extract_features_and_train
import os
import argparse


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
    extract_features_and_train(dirs, 3, 3,
                               0.2, 0.2, "svm_rbf",
                               output_model)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="the path of directory which "
                             "contains audio organized "
                             "in folders of classes")
    parser.add_argument("-o", "--outputmodelpath",
                        help="path to the final svm model to be saved")
    args = parser.parse_args()
    train_svm(args.input, args.outputmodelpath)
