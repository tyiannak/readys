import pickle5 as pickle
from recording_level_feature_extraction import RecordingLevelFeatureExtraction
import argparse
import yaml
import os


script_dir = os.path.dirname(__file__)
if not script_dir:
    with open(r'./config.yaml') as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
else:
    with open(script_dir + '/config.yaml') as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)

config = conf['recording_level_classifier']

def predict_recording_level_label(audio_file, model_path):
    '''

    :param audio_file: the path of audio file to test
    :param model_path: the path of the recording level model to use
    :return: class_name : the predicted class of the input file
    '''

    if not os.path.isfile(model_path):
        print("mtFileClassificationError: input model_type not found!")
        return
    basic_features_params = {}
    basic_features_params['google_credentials'] = config['google_credentials']
    basic_features_params['audio_models_folder'] = config['audio_models_folder']
    basic_features_params['text_models_folder'] = config['text_models_folder']

    model_dict = pickle.load(open(model_path, 'rb'))
    basic_features_params['features_type'] = model_dict['features_type']
    basic_features_params['reference_text'] = model_dict['reference_text']
    basic_features_params['text_segmentation_params'] = model_dict['text_segmentation_params']

    #feature_extraction
    feature_extractor = RecordingLevelFeatureExtraction(basic_features_params)
    if basic_features_params['reference_text']:
        folder = os.path.dirname(audio_file)
        file_name = os.path.basename(audio_file)
        file_name = os.path.splitext(file_name)[0]
        file_name = file_name + '.txt'
        textfile = [os.path.join(folder, file_name)]
    else:
        textfile = []
    feature_matrix , _ = feature_extractor.extract_recording_level_features([audio_file],textfile)

    classifier = model_dict['classifier']
    class_mapping = model_dict['class_mapping']
    class_id = classifier.predict(feature_matrix)
    class_id = int(class_id)
    class_name = class_mapping[class_id]
    return class_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",required=True,
                        help="the path of audio input file")
    parser.add_argument("-m", "--model_path", required=True,
                        help="the path of the model that we are gonna use to test")

    args = parser.parse_args()
    class_name = predict_recording_level_label(args.input,args.model_path)
    print(class_name)