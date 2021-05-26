"""
Combine functionality from ../text_analysis and ../audio_analysis
to extract recording-level aggregated feature extraction from both
audio and text modalities
"""

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path:
    sys.path.insert(0,parentdir)

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../'))

from text_analysis import get_asr_features
from audio_analysis import audio_based_feature_extraction
from models.utils import folders_mapping
import glob2 as glob
import numpy as np
from models.utils import load_classifiers


class RecordingLevelFeatureExtraction(object):
    def __init__(self, basic_features_params):
        """
            Initializes an RecordingLevelFeatureExtraction object by loading the
            basic feature extraction parameters
            :param basic_features_params: basic feature extraction parameters
        """
        self.basic_features_params = basic_features_params

    def fit(self):
        # comply with scikit-learn transformer requirement
        return self

    def transform(self, folder):
        # comply with scikit-learn transformer requirement
        """
        Extract features on a dataset which lies under the folder directory
        :param folder: the directory in which the dataset is located.
                        Each subfolder of the folder must contain instances
                        of a specific class.
        :return: 1. features: features (fused or audio or text) for each
                    dataset instance
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

        num_of_samples_per_class = []
        class_names = []
        if folders:
            for folder in folders:
                count = 0
                if reference_text:
                    for f in glob.iglob(os.path.join(folder, '*.txt')):
                        textnames.append(f)
                for f in glob.iglob(os.path.join(folder, '*.wav')):
                    filenames.append(f)
                    count += 1
                    labels.append(os.path.split(folder)[1])
                class_names.append(os.path.split(folder)[1])
                num_of_samples_per_class.append(count)
            folder2idx, idx2folder = folders_mapping(folders=folder_names)
            labels = list(map(lambda x: folder2idx[x], labels))
            labels = np.asarray(labels)

        else:
            filenames = [folder]
        print(labels)
        print(idx2folder)
        # Match filenames with labels
        print("class names",class_names)
        features, feature_names = \
            self.extract_recording_level_features(filenames,textnames)
        feature_list = []
        index = 0
        for num_of_samples in num_of_samples_per_class:
            class_features = features[index:index+num_of_samples]
            class_features = np.asarray(class_features)
            feature_list.append(class_features)
            index += num_of_samples

        print(feature_names)
        features = np.asarray(features)

        return features, labels, idx2folder, feature_list, feature_names, class_names, filenames

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
        audio_models_directory = \
            self.basic_features_params['audio_models_folder']
        text_models_directory = self.basic_features_params['text_models_folder']
        google_credentials = self.basic_features_params['google_credentials']
        segmentation_threshold = \
            self.basic_features_params['text_segmentation_params']['segmentation_threshold']
        method = self.basic_features_params['text_segmentation_params']['method_of_segmentation']
        overall_features = []

        # load text classifiers attributes containing embeddings
        # in order not to be loaded for every sample
        if features_type == "fused" or features_type == "text":
            classifiers_attributes = load_classifiers(text_models_directory)

        for count, file in enumerate(filenames):
            if textnames == []:
                reference_text = None
            else:
                reference_text = textnames[count]
            if features_type == "fused":

                if self.basic_features_params['audio_features'] == "fused":
                    audio_features, audio_features_names, _ = \
                        audio_based_feature_extraction(
                            file, audio_models_directory, mode=1,
                            pyaudio_params=self.basic_features_params['pyaudio_params'])
                elif self.basic_features_params['audio_features'] == "pyaudio":
                    audio_features, audio_features_names, _ = \
                        audio_based_feature_extraction(
                            file, audio_models_directory, mode=2,
                            pyaudio_params=self.basic_features_params['pyaudio_params'])
                else:
                    audio_features, audio_features_names, _ = \
                        audio_based_feature_extraction(
                            file, audio_models_directory, mode=0)
                text_features, text_features_names, _ = \
                    get_asr_features(file, google_credentials,
                                     classifiers_attributes, reference_text,
                                     segmentation_threshold, method)

                file_recording_level_features = audio_features + text_features
                file_features_names = audio_features_names + text_features_names
            elif features_type == "audio":
                if self.basic_features_params['audio_features'] == "fused":
                    audio_features, file_features_names, _ = \
                        audio_based_feature_extraction(
                            file, audio_models_directory, mode=1,
                            pyaudio_params=self.basic_features_params['pyaudio_params'])
                elif self.basic_features_params['audio_features'] == "pyaudio":
                    audio_features, file_features_names, _ = \
                        audio_based_feature_extraction(
                            file, audio_models_directory, mode=2,
                            pyaudio_params=self.basic_features_params['pyaudio_params'])
                else:
                    audio_features, file_features_names, _ = \
                        audio_based_feature_extraction(
                            file, audio_models_directory, mode=0)

                file_recording_level_features = audio_features
            elif features_type == "text":
                # load text classifiers attributes containing embeddings
                # in order not to be loaded for every sample
                file_recording_level_features, file_features_names, _ = \
                    get_asr_features(file, google_credentials,
                                     classifiers_attributes, reference_text,
                                     segmentation_threshold, method)
            file_recording_level_features = \
                np.asarray(file_recording_level_features)
            overall_features.append(file_recording_level_features)
        return overall_features , file_features_names
