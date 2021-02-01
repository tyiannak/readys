import os
import argparse
import yaml
import numpy as np
from feature_extraction import AudioFeatureExtraction


script_dir = os.path.dirname(__file__)
eps = np.finfo(float).eps
seed = 500
if not script_dir:
    with open(r'./config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
else:
    with open(script_dir + '/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)


def train_svm(files_path, output_model):
    """
    Train audio models
    :param files_path: directory which contains
        audio organized in folders of classes
    :param output_model: path to save the output model
    :return: the name of the saved model
    """
    config = config['audio_classifier']
    feature_extractor = AudioFeatureExtraction(config)
    #dirs = [x[0] for x in os.walk(files_path)]
    #dirs = sorted(dirs[1:])

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", required=True,
                        help="the path of directory which "
                             "contains audio organized "
                             "in folders of classes")
    parser.add_argument("-o", "--outputmodelpath", required=True,
                        help="path to the final svm model to be saved")
    args = parser.parse_args()
    if os.path.exists(args.input_folder) is False:
        raise FileNotFoundError()

    train_svm(args.input_folder, args.outputmodelpath)
