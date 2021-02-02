import os
import argparse
import yaml
import numpy as np
from feature_extraction import AudioFeatureExtraction
from utils import save_model, check_balance, train_basic_segment_classifier


script_dir = os.path.dirname(__file__)
eps = np.finfo(float).eps
seed = 500
if not script_dir:
    with open(r'./config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
else:
    with open(script_dir + '/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

config = config['audio_classifier']


def basic_segment_classifier(dir, out_model):
    """
    Train audio models
    :param files_path: directory which contains
        audio organized in folders of classes
    :param output_model: path to save the output model
    :return: the name of the saved model
    """
    basic_features_params = config['basic_features_params']
    feature_extractor = AudioFeatureExtraction(basic_features_params)
    features, labels, class_mapping = feature_extractor.transform(dir)

    is_imbalanced = check_balance(labels)
    clf = train_basic_segment_classifier(features, labels, is_imbalanced, config, seed)

    model_dict = {}
    model_dict['classifier_type'] = 'basic'
    model_dict['class_mapping'] = class_mapping
    model_dict['classifier'] = clf
    model_dict['basic_features_params'] = basic_features_params

    if out_model is None:
        save_model(model_dict, name="basic_classifier", is_text=False)
    else:
        save_model(model_dict, out_model=out_model, is_text=False)
    return None


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

    basic_segment_classifier(args.input_folder, args.outputmodelpath)