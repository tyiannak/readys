import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path:
    sys.path.insert(0,parentdir)
from text_analysis import get_asr_features
from audio_analysis import audio_based_feature_extraction
from utils import folders_mapping
import glob2 as glob
import numpy as np
from utils import test_if_already_loaded


class RecordingLevelFeatureExtraction(object):
    def __init__(self, basic_features_params):
        """
            Initializes an RecordingLevelFeatureExtraction object by loading the
            basic feature extraction parameters
            :param basic_features_params: basic feature extraction parameters
        """
        self.basic_features_params = basic_features_params
    def fit(self):  # comply with scikit-learn transformer requirement
        return self
    def transform(self, folder):  # comply with scikit-learn transformer requirement
        """
        Extract features on a dataset which lies under the folder directory
        :param folder: the directory in which the dataset is located.
                        Each subfolder of the folder must contain instances
                        of a specific class.
        :return: 1. features: features (fused or audio or text) for each dataset instance
                 2. labels: list of labels
                 3. idx2folder: a mapping from label numbers to label names
        """
        print("--> Extracting recording level features")
        filenames = []
        labels = []
        textnames = []
        folders = [x[0] for x in os.walk(folder)]
        folders = sorted(folders[1:])
        folder_names = [os.path.split(folder)[1] for folder in folders]
        reference_text = self.basic_features_params['reference_text']
        if folders:
            for folder in folders:
                if reference_text:
                    for f in glob.iglob(os.path.join(folder, '*.txt')):
                        textnames.append(f)
                for f in glob.iglob(os.path.join(folder, '*.wav')):
                    filenames.append(f)
                    labels.append(os.path.split(folder)[1])
            folder2idx, idx2folder = folders_mapping(folders=folder_names)
            labels = list(map(lambda x: folder2idx[x], labels))
            labels = np.asarray(labels)

        else:
            filenames = [folder]
        print(labels)
        print(idx2folder)
        # Match filenames with labels

        features, feature_names = \
            self.extract_recording_level_features(filenames,textnames)
        print(feature_names)
        features = np.asarray(features)

        return features, labels, idx2folder
    def extract_recording_level_features(self, filenames,textnames):
        """
        Extract unique overall files' features

        Parameters
        ----------

        filenames :
            List of input audio filenames

        textnames :
            List of input reference text filenames
        Returns
        -------

        overall_features:
            List of files' features
        file_features_names:
            List of feature names

        """
        features_type = self.basic_features_params['features_type']
        audio_models_directory = self.basic_features_params['audio_models_folder']
        audio_models_directory = audio_models_directory
        text_models_directory = self.basic_features_params['text_models_folder']
        text_models_directory =  text_models_directory
        google_credentials = self.basic_features_params['google_credentials']
        segmentation_threshold = self.basic_features_params['text_segmentation_params']['segmentation_threshold']
        method = self.basic_features_params['text_segmentation_params']['method_of_segmentation']
        overall_features = []

        #load text classifiers attributes containing embeddings in order not to be loaded for every sample
        classifiers_attributes = []
        for filename in os.listdir(text_models_directory):
            if filename.endswith(".pt"):
                model_path = os.path.join(text_models_directory, filename)
                dictionary = {}
                dictionary['classifier'], dictionary['classes'],dictionary['pretrained_path'], dictionary['pretrained'], \
                                dictionary['embeddings_limit'],dictionary['fasttext_model_path'] = \
                                test_if_already_loaded(model_path,classifiers_attributes)
                classifiers_attributes.append(dictionary)
        for count,file in enumerate(filenames):
            if textnames == []:
                reference_text = None
            else:
                reference_text = textnames[count]
            if features_type == "fused":
                audio_features, audio_features_names, _ = audio_based_feature_extraction(file,
                                                                                         audio_models_directory)
                text_features, text_features_names, _ = get_asr_features(file, google_credentials,
                                                                         classifiers_attributes, reference_text,
                                                                         segmentation_threshold, method)

                file_recording_level_features = audio_features + text_features
                file_features_names = audio_features_names + text_features_names
            elif features_type == "audio":
                file_recording_level_features, file_features_names, _ = audio_based_feature_extraction(file,
                                                                                             audio_models_directory)
            elif features_type == "text":
                file_recording_level_features, file_features_names, _ = get_asr_features(file, google_credentials,
                                                                               classifiers_attributes, reference_text,
                                                                               segmentation_threshold, method)
            file_recording_level_features = np.asarray(file_recording_level_features)
            overall_features.append(file_recording_level_features)
        return overall_features , file_features_names
