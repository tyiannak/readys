"""
This script is used to train and validate recording-level classifiers
"""

import os
import argparse
import yaml
from recording_level_feature_extraction import RecordingLevelFeatureExtraction
from utils import check_balance,train_recording_level_classifier,save_model,plot_feature_histograms


script_dir = os.path.dirname(__file__)
if not script_dir:
    with open(r'./config.yaml') as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
else:
    with open(script_dir + '/config.yaml') as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)

config = conf['recording_level_classifier']
seed = 500


def recording_level_classifier(inputs_path,model_name):
    '''
    Train recording level classifier using audio and text features
    :param inputs_path: the directory where samples are divided into
    class-folders
    :param model_name: the name of the model that we are gonna train
    :return: None. The classifier will be saved at the directory
    which is declared at the config.yaml file and with the input name.
    '''
    basic_features_params = config
    feature_extractor = RecordingLevelFeatureExtraction(basic_features_params)
    features, labels, class_mapping, feature_list, feature_names, class_names = feature_extractor.transform(inputs_path)

    plot_feature_histograms(feature_list, feature_names, class_names)

    is_imbalanced = check_balance(labels)

    clf = train_recording_level_classifier(features, labels, is_imbalanced, config, seed)
    model_dict = {}
    model_dict['classifier_type'] = config['classifier_type']
    model_dict['class_mapping'] = class_mapping
    model_dict['classifier'] = clf
    model_dict['features_type'] = config['features_type']
    model_dict['reference_text'] = config['reference_text']
    model_dict['text_segmentation_params'] = config['text_segmentation_params']
    model_dict['audio_features'] = config['audio_features']
    model_dict['pyaudio_params'] = config['pyaudio_params']
    out_folder = config['out_folder']
    if model_name is None:
        save_model(model_dict,out_folder, name="basic_classifier")
    else:
        save_model(model_dict,out_folder, out_model=model_name)
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",required=True,
                        help="the directory where samples "
                             "are devided into class folders")
    parser.add_argument("-mn","--model_name", required=True,
                        help="the name of the model "
                             "that we are gonna train")
    args = parser.parse_args()
    if os.path.exists(args.input) is False:
        raise FileNotFoundError()
    recording_level_classifier(args.input, args.model_name)
