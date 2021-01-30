import os
import numpy as np
import glob
from pyAudioAnalysis.audioTrainTest import write_train_data_arff,evaluate_classifier,normalize_features,train_svm,train_extra_trees,train_random_forest,train_knn,train_gradient_boosting,save_parameters,features_to_matrix
import pickle as cPickle
import argparse
from text_analysis import get_asr_features
from text_analysis import load_text_embedding_model as load_tem
from audio_analysis import audio_based_feature_extraction

def recording_level_feature_extraction(input_file,features_type,google_credentials = None,audio_models_directory = None,text_models_directory = None,
                                       reference_text=None,segmentation_threshold=None,method=None):
    '''
    Extract features from single audio file (either audio or text or fused)
    :param input_file: wav file from which features will be extracted
    :param features_type:
        -"fused" : for audio + text recording level features concatenated
        -"audio" : only audio recording level features
        -"text"  : only text recording level features
    :param google_credentials: the json file which contains google credentials (None if it is not needed)
    :param audio_models_directory: the directory of audio models
    :param text_models_directory: the directory of text models
    :param reference_text:
         - None : if no reference text is used else:
         - string of reference text
    :param segmentation_threshold: the duration or magnitude of every segment (for example: 2sec window or 2 words per segment)
    :param method:
        -None: the text will be segmented into sentences based on the punctuation that asr has found
        -"fixed_size_text" : split text into fixed size segments (fixed number of words)
        -"fixed_window" : split text into fixed time windows (fixed seconds)
    :return: recording level features , features' names
    '''
    if features_type == "fused" :
        audio_features, audio_features_names, _ = audio_based_feature_extraction(input_file,audio_models_directory)
        text_features, text_features_names, _ = get_asr_features(input_file, google_credentials,
                                                                 text_models_directory,reference_text,
                                                                 segmentation_threshold,method)

        recording_level_features = audio_features + text_features
        features_names = audio_features_names + text_features_names
    elif features_type == "audio":
        recording_level_features, features_names, _ = audio_based_feature_extraction(input_file,audio_models_directory)
    elif features_type == "text" :
        recording_level_features, features_names, _ = get_asr_features(input_file, google_credentials,
                                                                       text_models_directory, reference_text,
                                                                       segmentation_threshold, method)
    return recording_level_features, features_names

def train_recording_level_classifier(inputs_path,features_type,classifier_type, model_name,google_credentials = None,
                                     audio_models_directory=None,text_models_directory=None,reference_text=None,
                                     segmentation_threshold=None,method=None,train_percentage=0.90):
    '''
    Feature exraction for recording level classifier (either audio or text or fused) + train linear classifier.
    :param inputs_path: the directory where samples are devided into class folders
    :param features_type:
        -"fused" : for audio + text recording level features concatenated
        -"audio" : only audio recording level features
        -"text"  : only text recording level features
    :param classifier_type: the linea classifier that we want to train
        - "svm"
        - "svm_rbf"
        - "randomforest"
        - "knn"
        - "gradientboosting"
        - "extratrees"
    :param model_name: the name of the model that we are going to train
    :param google_credentials: the json file which contains google credentials (None if it is not needed)
    :param audio_models_directory: the directory of audio models
    :param text_models_directory: the directory of text models
    :param reference_text:
        - None : if no reference text is used otherwise:
        - the directory where reference texts (txt filed) are devided into class folders
    :param segmentation_threshold: the duration or magnitude of every segment (for example: 2sec window or 2 words per segment)
    :param method:
        -None: the text will be segmented into sentences based on the punctuation that asr has found
        -"fixed_size_text" : split text into fixed size segments (fixed number of words)
        -"fixed_window" : split text into fixed time windows (fixed seconds)
    :param train_percentage: the percentage of the dataset that will be uses as train set (the remaining will be used as validation set)
    :return: nothing to return. the model file is saved with model_name as well as the model parameters with name model_name + "MEANS"
    '''
    if reference_text:
        text_path_list =  [x[0] for x in os.walk(reference_text)]
        text_path_list = sorted(text_path_list[1:])
    path_list = [x[0] for x in os.walk(inputs_path)]
    path_list = sorted(path_list[1:])
    types = ('*.wav', '*.aif', '*.aiff', '*.mp3', '*.au', '*.ogg')
    text_types = ('*.txt')
    features = []
    class_names = []
    #iterate through class directories
    for i, folder_path in enumerate(path_list):
        wav_file_list= []
        #find wav files for the specific directory-class
        for files in types:
            wav_file_list.extend(glob.glob(os.path.join(folder_path, files)))
        if reference_text:
            text_file_list = []
            for files in text_types:
                text_file_list.extend(glob.glob(os.path.join(text_path_list[i], files)))
        directory_features = np.array([])
        #iterate through wav files
        for i, file_path in enumerate(wav_file_list):
            if reference_text:
                with open(text_file_list[i], 'r') as file:
                    reference_text = file.read()
            print("Analyzing file {0:d} of {1:d}: {2:s}".format(i + 1,
                                                                len(wav_file_list),
                                                                file_path))
            if os.stat(file_path).st_size == 0:
                print("   (EMPTY FILE -- SKIPPING)")
                continue
            #feature extraction for every wav file
            file_features,file_features_names = recording_level_feature_extraction(file_path,features_type,google_credentials,audio_models_directory,
                                                                                   text_models_directory,reference_text,
                                                                                   segmentation_threshold,method)
            file_features = np.array(file_features)
            if len(directory_features) == 0:
                # append directory features (features of a specific class)
                directory_features = file_features
            else:
                directory_features = np.vstack((directory_features, file_features))
        if directory_features.shape[0] > 0:
            # if at least one audio file has been found in the provided folder:
            #append overall features (features of all classes)
            features.append(directory_features)
            if folder_path[-1] == os.sep:
                class_names.append(folder_path.split(os.sep)[-2])
            else:
                class_names.append(folder_path.split(os.sep)[-1])
    if len(features) == 0:
        print("trainSVM_feature ERROR: No data found in any input folder!")
        return
    n_feats = features[0].shape[1]
    feature_names = ["features" + str(d + 1) for d in range(n_feats)]

    write_train_data_arff(model_name, features, class_names, feature_names)
    for i, feat in enumerate(features):
        if len(feat) == 0:
            print("trainSVM_feature ERROR: " + path_list[i] +
                  " folder is empty or non-existing!")
            return
    # STEP B: classifier Evaluation and Parameter Selection:
    if classifier_type == "svm" or classifier_type == "svm_rbf":
        classifier_par = np.array([0.001, 0.01, 0.5, 1.0, 5.0, 10.0, 20.0])
    elif classifier_type == "randomforest":
        classifier_par = np.array([10, 25, 50, 100, 200, 500])
    elif classifier_type == "knn":
        classifier_par = np.array([1, 3, 5, 7, 9, 11, 13, 15])
    elif classifier_type == "gradientboosting":
        classifier_par = np.array([10, 25, 50, 100, 200, 500])
    elif classifier_type == "extratrees":
        classifier_par = np.array([10, 25, 50, 100, 200, 500])
    best_param = evaluate_classifier(features, class_names, 100, classifier_type,
                                     classifier_par, 0, train_percentage)

    print("Selected params: {0:.5f}".format(best_param))

    features_norm, mean, std = normalize_features(features)
    mean = mean.tolist()
    std = std.tolist()

    # STEP C: Save the classifier to file
    if classifier_type == "svm":
        classifier = train_svm(features_norm, best_param)
    elif classifier_type == "svm_rbf":
        classifier = train_svm(features_norm, best_param, kernel='rbf')
    elif classifier_type == "randomforest":
        classifier = train_random_forest(features_norm, best_param)
    elif classifier_type == "gradientboosting":
        classifier = train_gradient_boosting(features_norm, best_param)
    elif classifier_type == "extratrees":
        classifier = train_extra_trees(features_norm, best_param)

    if classifier_type == "knn":
        feature_matrix, labels = features_to_matrix(features_norm)
        feature_matrix = feature_matrix.tolist()
        labels = labels.tolist()
        save_path = model_name
        save_parameters(save_path, feature_matrix, labels, mean, std,
                        class_names, best_param,features_type,segmentation_threshold,method)
    elif classifier_type == "svm" or classifier_type == "svm_rbf" or \
            classifier_type == "randomforest" or \
            classifier_type == "gradientboosting" or \
            classifier_type == "extratrees":
        with open(model_name, 'wb') as fid:
            cPickle.dump(classifier, fid)
        save_path = model_name + "MEANS"
        save_parameters(save_path, mean, std, class_names,features_type,segmentation_threshold,method)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",required=True,
                        help="the directory where samples are devided into class folders")
    parser.add_argument("-f","--fusion",required=True,
                        help=" fused, audio or text for feature extraction")
    parser.add_argument("-ct","--classifier_type",required = True,
                        help = "svm or svm_rbf or knn or gradientboosting or randomforest or extratrees")
    parser.add_argument("-mn","--model_name",required= True,
                        help = "the name of the model that we are gonna train")
    parser.add_argument("-g", "--google_credentials", required=False, default = None,
                        help=".json file with google credentials")
    parser.add_argument("-a", "--audio_models_path",required=False,default=None,
                        help="the directory which contains all trained audio classifiers (models' files + MEANS files)")
    parser.add_argument("-t", "--text_models_path", required=False,default=None,
                        help="the directory which contains all trained text classifiers (models' files + .csv classes_names files)")
    parser.add_argument('-r', '--reference_text', required=False, default=None,
                        help='None for no reference text or path which contains reference texts devided into class directories')
    parser.add_argument('-s', '--segmentation_threshold', required=False, default=None, type=int,
                        help='number of words or seconds of every text segment')
    parser.add_argument('-m', '--method_of_segmentation', required=False, default=None,
                        help='Choice between "fixed_size_text" and "fixed_window"')
    parser.add_argument('-tp','--train_percentage',required = False, default = 0.9 ,type = float,
                        help = "the percentage of the dataset that will be used as train set")
    args = parser.parse_args()

    if args.fusion == "fused" and (args.audio_models_path == None or args.text_models_path == None or args.google_credentials == None):
        print("Error, you need to input both audio models directory and text models directory as well as embedding model path and google credentials for fused feature extraction")
    elif args.fusion == "audio" and args.audio_models_path == None :
        print("Error, you need to input audio models directory for audio feature extraction ")
    elif args.fusion == "text" and (args.text_models_path == None or args.google_credentials == None):
        print("Error, you need to input text models directory as well as embedding model path and google credentials for text feature extraction")
    else:
        train_recording_level_classifier(args.input,args.fusion,args.classifier_type,args.model_name,args.google_credentials,args.audio_models_path,
                                         args.text_models_path,args.reference_text,
                                         args.segmentation_threshold,args.method_of_segmentation,args.train_percentage)
