"""
This script is used to train and validate recording-level classifiers
"""

import os
import argparse
import yaml
from recording_level_feature_extraction import RecordingLevelFeatureExtraction
from utils import check_balance,train_recording_level_classifier,save_model,plot_feature_histograms,train_late_fusion


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
    fused_features, feature_list, fused_names, readys_features, \
    readys_list, readys_names, pyaudio_features, pyaudio_list, \
    pyaudio_names, labels, class_mapping, class_names, filenames = feature_extractor.transform(inputs_path)

    is_imbalanced = check_balance(labels)
    if basic_features_params['audio_features'] == "late_fused":
        filename = "readys_report.html"
        plot_feature_histograms(readys_list, readys_names,class_names,filename)
        filename = "pyaudio_report.html"
        plot_feature_histograms(pyaudio_list,pyaudio_names,class_names,filename)
        clf_readys,clf_pyaudio = train_late_fusion(readys_features,pyaudio_features,labels,is_imbalanced,config,filenames,seed)
        model_dict = {}
        model_dict['classifier_type'] = config['late_fusion']['classifier_pyaudio']
        model_dict['class_mapping'] = class_mapping
        model_dict['classifier'] = clf_pyaudio
        model_dict['features_type'] = config['features_type']
        model_dict['reference_text'] = config['reference_text']
        model_dict['text_segmentation_params'] = config['text_segmentation_params']
        model_dict['audio_features'] = config['audio_features']
        model_dict['pyaudio_params'] = config['pyaudio_params']
        model_dict['pyaudio_num_features'] = config['pyaudio_num_features']
        model_dict['raudio_num_features_discard'] = config['raudio_num_features_discard']
        model_dict['gender'] = config['gender']
        out_folder = config['out_folder']
        if model_name is None:
            save_model(model_dict, out_folder, name="basic_classifier_pyaudio")
        else:
            save_model(model_dict, out_folder, out_model=model_name +"_pyaudio")
        model_dict['classifier_type'] = config['late_fusion']['classifier_raudio']
        model_dict['classifier'] = clf_readys
        if model_name is None:
            save_model(model_dict, out_folder, name="basic_classifier_readys")
        else:
            save_model(model_dict, out_folder, out_model=model_name + "_readys")
    else:
        filename = "report.html"
        plot_feature_histograms(feature_list, fused_names,class_names,filename)
        clf = train_recording_level_classifier(fused_features, labels, is_imbalanced, config, filenames, seed)
        model_dict = {}
        model_dict['classifier_type'] = config['classifier_type']
        model_dict['class_mapping'] = class_mapping
        model_dict['classifier'] = clf
        model_dict['features_type'] = config['features_type']
        model_dict['reference_text'] = config['reference_text']
        model_dict['text_segmentation_params'] = config['text_segmentation_params']
        model_dict['audio_features'] = config['audio_features']
        model_dict['pyaudio_params'] = config['pyaudio_params']
        model_dict['pyaudio_num_features'] = config['pyaudio_num_features']
        model_dict['raudio_num_features_discard'] = config['raudio_num_features_discard']
        model_dict['gender'] = config['gender']
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
