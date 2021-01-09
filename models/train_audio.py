from pyAudioAnalysis.audioTrainTest import extract_features_and_train
import os
import argparse

def train_svm(files_path):
    '''
    Train audio models
    :param files_path: directory which contains audio organized in folders of classes
    :return: the name of the saved model
    '''
    onlydir = [x[0] for x in os.walk(files_path)]
    onlydir = onlydir[1:]
    extract_features_and_train(onlydir,3,3,0.2,0.2,"svm_rbf","valence_svm_rbf")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="the path of directory which contains audio organized in folders of classes" )
    args = parser.parse_args()
    train_svm(args.input)
