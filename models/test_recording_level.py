"""
This script is used to test a trained recording-level classifier
"""

import pickle5 as pickle
import sys
import os

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../'))

from models.recording_level_feature_extraction \
    import RecordingLevelFeatureExtraction,RecordingLevelFeatureLoading
import argparse
import yaml
import os
import numpy as np

script_dir = os.path.dirname(__file__)
if not script_dir:
    with open(r'./config.yaml') as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
else:
    with open(script_dir + '/config.yaml') as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)

config = conf['recording_level_classifier']

def predict_recording_level_label(audio_file, model1, model2=None):
    """
    :param audio_file: the path of audio file to test
    :param model_path: the path of the recording level model to use
    :return: class_name : the predicted class of the input file
    """
    basic_features_params = {}
    basic_features_params['google_credentials'] = config['google_credentials']
    basic_features_params['audio_models_folder'] = config['audio_models_folder']
    basic_features_params['text_models_folder'] = config['text_models_folder']


    model_dict = pickle.load(open(model1, 'rb'))

    basic_features_params['classifier_type'] = model_dict['classifier_type']
    basic_features_params['class_mapping'] = model_dict['class_mapping']
    basic_features_params['classifier'] = model_dict['classifier']
    basic_features_params['features_type']  = model_dict['features_type']
    basic_features_params['reference_text']  = model_dict['reference_text']
    basic_features_params['text_segmentation_params'] = model_dict['text_segmentation_params']
    basic_features_params['audio_features'] = model_dict['audio_features']
    basic_features_params['pyaudio_params'] = model_dict['pyaudio_params']
    basic_features_params['pyaudio_num_features'] = model_dict['pyaudio_num_features']
    basic_features_params['raudio_num_features_discard']  = model_dict['raudio_num_features_discard']
    basic_features_params['gender'] = model_dict['gender']


    if basic_features_params['reference_text']:
        folder = os.path.dirname(audio_file)
        file_name = os.path.basename(audio_file)
        file_name = os.path.splitext(file_name)[0]
        file_name = file_name + '.txt'
        textfile = [os.path.join(folder, file_name)]
    else:
        textfile = []
    if audio_file.endswith('.wav'):
        feature_extractor = RecordingLevelFeatureExtraction(basic_features_params)
        fused_features, fused_names, readys_features, readys_names, pyaudio_features, pyaudio_names, labels, filenames = \
            feature_extractor.extract_recording_level_features([audio_file], textfile, ['positive'])
    else:
        feature_extractor = RecordingLevelFeatureLoading(basic_features_params)
        fused_features, fused_names, readys_features, readys_names, pyaudio_features, pyaudio_names, labels, filenames= \
            feature_extractor.load_recording_level_features([audio_file],textfile,['positive'])
    print(readys_names)
    print(readys_features)
    print(pyaudio_names)
    print(pyaudio_features)
    classifier = model_dict['classifier']
    class_mapping = model_dict['class_mapping']
    if basic_features_params['audio_features'] == "fused":
        class_id = classifier.predict(fused_features)
        class_id = int(class_id)
        class_name = class_mapping[class_id]
    elif basic_features_params['audio_features'] == "audio":
        class_id = classifier.predict(readys_features)
        class_id = int(class_id)
        class_name = class_mapping[class_id]
    else:
        proba1 = classifier.predict_proba(readys_features)
        model_dict2 = pickle.load(open(model2, 'rb'))
        classifier2 = model_dict2['classifier']
        proba2 = classifier2.predict_proba(pyaudio_features)
        average = np.mean([proba1, proba2], axis=0)
        y_pred = np.argmax(average, axis=1)
        class_id = int(y_pred)
        class_name = class_mapping[class_id]
    print("class name:", class_name)
    print("class id:", class_id)
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",required=True,
                        help="the path of audio input file or MetaAudio feature file")
    parser.add_argument("-m", "--model_path", required=True,
                        help="the path of the model that "
                             "we are gonna use to test")
    parser.add_argument("-m2", "--model2_path", required=False, default='None',
                        help="the path of the pyaudio model if we use late fusion")

    args = parser.parse_args()
    predict_recording_level_label(args.input, args.model_path, args.model2_path)
