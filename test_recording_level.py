import os
import numpy as np
import pickle as cPickle
from train_recording_level_classifier import recording_level_feature_extraction
from pyAudioAnalysis.audioTrainTest import Knn
import argparse

def predict_recording_level_labels(audio_file, model_name,model_type,google_credentials,audio_models_directory,
                                   text_models_directory,reference_text=None):
    labels = []
    accuracy = 0.0
    class_names = []
    cm = np.array([])
    if not os.path.isfile(model_name):
        print("mtFileClassificationError: input model_type not found!")
        return labels, class_names, accuracy, cm

    # Load classifier:
    if model_type == "knn":
        with open(model_name, "rb") as fo:
            features = cPickle.load(fo)
            labels = cPickle.load(fo)
            mean = cPickle.load(fo)
            std = cPickle.load(fo)
            classNames = cPickle.load(fo)
            neighbors = cPickle.load(fo)
            features_type = cPickle.load(fo)
            segmentation_threshold = cPickle.load(fo)
            method = cPickle.load(fo)
        features = np.array(features)
        labels = np.array(labels)
        mean = np.array(mean)
        std = np.array(std)

        classifier = Knn(features, labels, neighbors)
    else:
        with open(model_name + "MEANS", "rb") as fo:
            mean = cPickle.load(fo)
            std = cPickle.load(fo)
            classNames = cPickle.load(fo)
            features_type = cPickle.load(fo)
            segmentation_threshold = cPickle.load(fo)
            method = cPickle.load(fo)
        mean = np.array(mean)
        std = np.array(std)
        with open(model_name, 'rb') as fid:
            classifier = cPickle.load(fid)

    #feature_extraction
    feature_matrix = recording_level_feature_extraction(audio_file,features_type,google_credentials,audio_models_directory,
                                                        text_models_directory,reference_text,
                                                        segmentation_threshold,method)
    normalized_features = (feature_matrix - mean)/std
    class_id = classifier.predict(normalized_features)
    class_name = classNames[class_id]
    return class_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",required=True,
                        help="the path of audio input file")
    parser.add_argument("-mn", "--model_name", required=True,
                        help="the name of the model that we are gonna use to test")
    parser.add_argument("-mt", "--model_type", required=True,
                        help="the type of the model that we are gonna use to test")
    parser.add_argument("-g", "--google_credentials", required=False, default=None,
                        help=".json file with google credentials")
    parser.add_argument("-a", "--audio_models_path", required=False, default=None,
                        help="the directory which contains all trained audio classifiers (models' files + MEANS files)")
    parser.add_argument("-t", "--text_models_path", required=False, default=None,
                        help="the directory which contains all trained text classifiers (models' files + .csv classes_names files)")
    parser.add_argument('-r', '--reference_text', required=False, default=None,
                        help='None for no reference text or path which contains reference texts devided into class directories')
    args = parser.parse_args()

    if args.fusion == "fused" and (args.audio_models_path == None or args.text_models_path == None or args.google_credentials == None):
        print("Error, you need to input both audio models directory and text models directory as well as embedding model path and google credentials for fused feature extraction")
    elif args.fusion == "audio" and args.audio_models_path == None :
        print("Error, you need to input audio models directory for audio feature extraction ")
    elif args.fusion == "text" and (args.text_models_path == None  or args.google_credentials == None):
        print("Error, you need to input text models directory as well as embedding model path and google credentials for text feature extraction")
    else:
       class_name = predict_recording_level_labels(args.input,args.model_name,args.model_type,args.google_credentials,
                                                   args.audio_models_path,args.text_models_path,args.reference_text)
       print(class_name)