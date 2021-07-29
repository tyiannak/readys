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
        if basic_features_params["gender"] == "male" or basic_features_params["gender"] == "female":
            self.gender = basic_features_params["gender"]
        else:
            self.gender = None
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
                        if self.gender is not None:
                            if "female" not in f:
                                if self.gender == "female":
                                    continue
                            elif self.gender == "male":
                                continue

                        textnames.append(f)
                for f in glob.iglob(os.path.join(folder, '*.wav')):
                    '''
                    if self.gender is not None:
                        if "female" not in f:
                            if self.gender == "female":
                                continue
                        elif self.gender == "male":
                            continue
                    '''
                    filenames.append(f)
                    count += 1
                    labels.append(os.path.split(folder)[1])
                class_names.append(os.path.split(folder)[1])
                num_of_samples_per_class.append(count)
            folder2idx, idx2folder = folders_mapping(folders=folder_names)
            labels = list(map(lambda x: folder2idx[x], labels))


        else:
            filenames = [folder]
        print(labels)
        print(idx2folder)
        # Match filenames with labels
        print("class names",class_names)
        fused_features, fused_names, readys_features, readys_names, pyaudio_features, pyaudio_names, labels, filenames = \
            self.extract_recording_level_features(filenames, textnames, labels)
        labels = np.asarray(labels)
        feature_list = []
        readys_list = []
        pyaudio_list = []
        if readys_features == []  and pyaudio_features == []:
            index = 0
            for num_of_samples in num_of_samples_per_class:
                class_features = fused_features[index:index+num_of_samples]
                class_features = np.asarray(class_features)
                feature_list.append(class_features)
                index += num_of_samples
            print(fused_names)
            fused_features = np.asarray(fused_features)
        else:
            index = 0
            for num_of_samples in num_of_samples_per_class:
                class_features_readys = readys_features[index:index+num_of_samples]
                class_features_readys = np.asarray(class_features_readys)
                readys_list.append(class_features_readys)
                class_features_pyaudio = pyaudio_features[index:index + num_of_samples]
                class_features_pyaudio = np.asarray(class_features_pyaudio)
                pyaudio_list.append(class_features_pyaudio)
                index += num_of_samples
            print("Readys features:",readys_names)
            readys_features = np.asarray(readys_features)
            print("Pyaudio features:", pyaudio_names)
            pyaudio_features = np.asarray(pyaudio_features)
        return fused_features, feature_list, fused_names, readys_features,\
               readys_list, readys_names, pyaudio_features, pyaudio_list,\
               pyaudio_names, labels, idx2folder, class_names, filenames

    def extract_recording_level_features(self, filenames, textnames, labels):
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
        overall_raudio_features = []
        overall_pyaudio_features = []
        # load text classifiers attributes containing embeddings
        # in order not to be loaded for every sample
        if features_type == "fused" or features_type == "text":
            classifiers_attributes = load_classifiers(text_models_directory)

        audio_features = []
        audio_features_names = []
        raudio_features_names = []
        pyaudio_features_names = []
        new_labels = []
        new_filenames = []
        for count, file in enumerate(filenames):
            text_not_exist = False
            '''
            if self.gender is not None:
                if "female" not in file:
                    if self.gender == "female":
                        continue
                elif self.gender == "male":
                    continue
            '''
            if textnames == []:
                reference_text = None
            else:
                reference_text = textnames[count]
            if features_type == "fused" or features_type == "audio":
                pyaudio_num_features = self.basic_features_params['pyaudio_num_features']
                raudio_features_discard = self.basic_features_params['raudio_num_features_discard']
                if self.basic_features_params['audio_features'] == "fused":
                    audio_features, audio_features_names, _ = \
                        audio_based_feature_extraction(
                            file, audio_models_directory,raudio_features_discard=raudio_features_discard,
                            pyaudio_num_features=pyaudio_num_features, mode=1,
                            pyaudio_params=self.basic_features_params['pyaudio_params'])
                elif self.basic_features_params['audio_features'] == "pyaudio":
                    audio_features, audio_features_names, _ = \
                        audio_based_feature_extraction(
                            file, audio_models_directory,pyaudio_num_features=pyaudio_num_features, mode=2,
                            pyaudio_params=self.basic_features_params['pyaudio_params'])
                elif self.basic_features_params['audio_features'] == "audio":
                    audio_features, audio_features_names, _ = \
                        audio_based_feature_extraction(
                            file, audio_models_directory,raudio_features_discard=raudio_features_discard,
                            mode=0)
                else:
                    raudio_features, raudio_features_names, _ = \
                        audio_based_feature_extraction(
                            file, audio_models_directory, raudio_features_discard=raudio_features_discard,
                            mode=0)
                    pyaudio_features, pyaudio_features_names, _ = \
                        audio_based_feature_extraction(
                            file, audio_models_directory, pyaudio_num_features=pyaudio_num_features, mode=2,
                            pyaudio_params=self.basic_features_params['pyaudio_params'])
                if features_type == "fused":
                    text_features, text_features_names, _ = \
                        get_asr_features(file, google_credentials,
                                         classifiers_attributes, reference_text,
                                         segmentation_threshold, method)
                    if text_features == []:
                        text_not_exist = True
                    if self.basic_features_params['audio_features'] == "late_fused":
                        raudio_features += text_features
                        raudio_features_names += text_features_names
                    else:
                        audio_features += text_features
                        audio_features_names += text_features_names
                file_recording_level_features = audio_features
                file_features_names = audio_features_names
            elif features_type == "text":
                # load text classifiers attributes containing embeddings
                # in order not to be loaded for every sample
                file_recording_level_features, file_features_names, _ = \
                    get_asr_features(file, google_credentials,
                                     classifiers_attributes, reference_text,
                                     segmentation_threshold, method)
                if file_recording_level_features == []:
                    text_not_exist = True
            if self.basic_features_params['audio_features'] == "late_fused":
                raudio_features = np.asarray(raudio_features)
                pyaudio_features = np.asarray(pyaudio_features)
                if text_not_exist == False:
                    overall_raudio_features.append(raudio_features)
                    overall_pyaudio_features.append(pyaudio_features)
                    new_labels.append(labels[count])
                    new_filenames.append(file)
            else:
                file_recording_level_features = \
                    np.asarray(file_recording_level_features)
                if text_not_exist == False:
                    overall_features.append(file_recording_level_features)
                    new_labels.append(labels[count])
                    new_filenames.append(file)
        return overall_features , file_features_names, overall_raudio_features, raudio_features_names, overall_pyaudio_features, pyaudio_features_names, new_labels, new_filenames
