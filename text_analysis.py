"""
Given an audio file this module is capable of :
 - using asr.audio_to_asr_text() to transcode speech to text (using google api)
 - extracting aggregates of text features (text_features()) using
   models.test_text.predict() for all available segment text models
 - extracting text reference features if available
 - merging the above in a recording-level text representation
"""

import asr
import text_scoring as ts
import numpy as np
from models.test_text import predict
import argparse
import re
import os
from models.utils import load_classifiers
from pathlib import Path
import pickle5 as pickle


def load_reference_data(path):
    text = open(path).read()
    return text


def text_preprocess(document):
    """
    Basic text preprocessing
    :param document: string containing input text
    :return: updated text
    """
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))
    # Substitute multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Convert to lowercase
    document = document.lower()
    return document


def text_segmentation(text, segmentation_threshold=None,
                      method=None, asr_timestamps=None):
    """
    Break text into segments in accordance with a defined method
    :param text: the text to be segmented
    :param segmentation_threshold: the duration or magnitude of every segment
           (for example: 2sec window or 2 words per segment)
    :param method:
    -None: the text will be segmented into sentences based on the punctuation
           that asr has found
    -"fixed_size_text": split text into fixed size segments
                        (fixed number of words)
    -"fixed_window": split text into fixed time windows (fixed seconds)
    :param asr_timestamps: the timestamps of words that asr has defined
    :return:
      -text_segmented : a list of segments of the text
                       (every element of the list is a string)
    """

    if method == 'None' or method == None:
        text_segmented = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s',
                                  text)
    elif method == "fixed_size_text":
        text = text_preprocess(text)
        words = text.split()
        text_segmented = []
        for i in range(0, len(words), segmentation_threshold):
            text_segmented.append(" ".join(words[i:i + segmentation_threshold]))
    elif method == "fixed_window":
        first_word =asr_timestamps[0]
        start_time = first_word['st']
        last_word = asr_timestamps[-1]
        end_time = last_word['et']
        start_time_of_window = start_time
        cur_segment = ""
        text_segmented = []
        iter_of_words = 0
        word = asr_timestamps[iter_of_words]
        #iterate through time windows
        while start_time_of_window < end_time:
            #iterate through timestamps
            #if the word is included in the time window thw while is activated
            while word['st'] >= start_time_of_window and word['st'] <= \
                    (start_time_of_window + segmentation_threshold):
                #save string of the current segment
                if cur_segment == "":
                    cur_segment = word['word']
                else:
                    cur_segment = cur_segment + " " + word['word']
                #if we haven' t reached the last word, continue else break
                if iter_of_words < (len(asr_timestamps) - 1):
                    iter_of_words += 1
                    word = asr_timestamps[iter_of_words]
                else:
                    break
            #update list of segments
            text_segmented.append(cur_segment)
            cur_segment = ""
            start_time_of_window += segmentation_threshold
    return text_segmented


def text_features(text, classifiers_attributes, segmentation_threshold=None,
                  method=None, asr_results=None):
    '''
    Features exported from models(classifiers)
    :param text: the text we want to extract features from (string)
    :classifiers_attributes: a list of dictionaries with keys :
    classifier,classes,pretrained_path,pretrained,embeddings_limit,
    fasttext_model_path.
     Every dictionary refers to a classifier previously loaded.
    :param segmentation_threshold: the duration or magnitude of every segment
    (for example: 2sec window or 2 words per segment)
    :param method:
    -None: the text will be segmented into sentences based on the
       punctuation that asr has found
    -"fixed_size_text" : split text into fixed size segments
      (fixed number of words)
    -"fixed_window" : split text into fixed time windows (fixed seconds)
    :param asr_results: the timestamps of words that asr has defined
    :return:
    - features: list of text features extracted
    - features_names: list of respective feature names
    '''
    features = []
    features_names = []

    # TODO: load all segment-level models that have been trainied in
    #       a predefined path such as segment_models/text
    # TODO: add pretrained model posteriors, e.g. P(y=negative|x) etc
    dictionaries = []
    text_segmented = text_segmentation(text, segmentation_threshold, method,
                                       asr_results)
    print(text_segmented)

    #for every text classifier (with embeddings already loaded)
    for classifier_dictionary in classifiers_attributes:
        classifier, classes, pretrained, embeddings_limit = \
            classifier_dictionary['classifier'],\
            classifier_dictionary['classes'],\
            classifier_dictionary['pretrained'],\
            classifier_dictionary['embeddings_limit']
        dictionary , _ = predict(text_segmented, classifier, classes,
                                 pretrained, embeddings_limit)
        dictionaries.append(dictionary)
    for dictionary in dictionaries:
        for label in dictionary:
            feature_string = label + "(%)"
            feature_value = dictionary[label]
            features_names.append(feature_string)
            features.append(feature_value)
    return features, features_names


def get_asr_features(input_file, google_credentials,
                     classifiers_attributes, reference_text=None,
                     segmentation_threshold=None, method=None):
    """
    Extract text features from ASR results of a speech audio file
    :param input_file: path to the audio file
    :param google_credentials: path to the ASR google credentials file
    :classifiers_attributes: a list of dictionaries with keys : classifier,
                             classes, pretrained_path,pretrained,
                             embeddings_limit, fasttext_model_path.
     Every dictionary refers to a classifier previously loaded.
    :param reference_text:  path to the reference text
    :param segmentation_threshold: the duration or magnitude of every segment
                            (for example: 2sec window or 2 words per segment)
    :param method:
    - None: the text will be segmented into sentences based on the punctuation
    that asr has found
    - "fixed_size_text" : split text into fixed size segments (fixed number of
    words)
    - "fixed_window" : split text into fixed time windows (fixed seconds)
    :return:
     - features: list of text features extracted
     - feature_names: list of respective feature names
     - metadata: list of metadata
    """

    feature_names = []
    features = []
    # Step 1: speech recognition using google speech API:
    # check if asr file already exists
    folder = os.path.dirname(input_file)
    file_name = os.path.basename(input_file)
    file_name = os.path.splitext(file_name)[0]
    file_name = file_name +  '.asr'
    full_path = os.path.join(folder, file_name)
    full_path = Path(full_path)
    if full_path.is_file():
        # loading asr from cache
        print("--> Loading saved asr")
        asr_dict = pickle.load(open(full_path, 'rb'))
        asr_results = asr_dict['timestamps']
        data = asr_dict['text']
        n_words = asr_dict['n_words']
        dur = asr_dict['dur']
    else:
        print("--> Audio to asr text via google speech Api")
        asr_results, data, n_words, dur = \
            asr.audio_to_asr_text(input_file,  google_credentials)
        asr_dict = {}
        asr_dict['timestamps'] = asr_results
        asr_dict['text'] = data
        asr_dict['n_words'] = n_words
        asr_dict['dur'] = dur
        # caching asr results
        with open(full_path, 'wb') as handle:
            pickle.dump(asr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(asr_results)
    print(data)
    # Step 2: compute basic text features and metadata:
    word_rate = float("{:.2f}".format(n_words / (dur / 60.0)))
    metadata = {"asr timestamps": asr_results,
                "Number of words": n_words,
                "Total duration (sec)": dur}

    # Step 3: compute reference text - related features
    # (if reference text is available)
    if reference_text:
        # get the reference text and align with the predicted text
        # (word to word alignment):
        ref_text = load_reference_data(reference_text)
        alignment, rec, pre = ts.text_to_text_alignment_and_score(ref_text,
                                                                  data)
        # get the f1 (recall / precision are computed between the
        # reference_text and the predicted text
        f1 = 2 * rec * pre / (rec + pre + np.finfo(np.float32).eps)

        rec = float("{:.2f}".format(rec))
        pre = float("{:.2f}".format(pre))
        f1 = float("{:.2f}".format(f1))

        feature_names = ["Recall score (%)",
                         "Precision score(%)",
                         "F1 score (%)"]
        features = [rec, pre, f1]

        # temporal score calculation:
        # (this info is used ONLY for plotting, so it is returned as metadata)
        if alignment != []:
            adjusted_results = ts.adjust_asr_results(asr_results,
                                                     alignment.second.elements,
                                                     dur)
            length = 0.5
            step = 0.1
            recalls, precisions, f1s, ref, asr_r = \
                ts.windows(alignment.first.elements, alignment.second.elements,
                           adjusted_results, length, step, dur)
        else:
            length = 0.5
            step = 0.1
            i=length
            recalls = []
            precisions = []
            f1s = []
            total_number_of_windows = 0
            while (i + length )< dur:
                total_number_of_windows += 1
                recalls.append({"x": i, "y": 0})
                precisions.append({"x": i, "y": 0})
                f1s.append({"x": i, "y": 0})
                i += step
            ref, asr_r = ["-"] * total_number_of_windows,\
                         ["-"] * total_number_of_windows
        metadata["temporal_recall"] = recalls
        metadata["temporal_precision"] = precisions
        metadata["temporal_f1"] = f1s
        metadata["temporal_ref"] = ref
        metadata["temporal_asr"] = asr_r


    feature_names.append("Word rate (words/min)")
    features.append(word_rate)

    # Pure-text-based features:

    features_text, features_names_text = text_features(data,
                                                       classifiers_attributes,
                                                       segmentation_threshold,
                                                       method,
                                                       asr_results)

    features += features_text
    feature_names += features_names_text

    return features, feature_names, metadata

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",required=True,
                        help="path of wav file")
    parser.add_argument("-g", "--google_credentials",required=True,
                        help=".json file with google credentials")
    parser.add_argument("-c", "--classifiers_path",required=True,
                        help="the directory which contains "
                             "all text trained classifiers")
    parser.add_argument('-r', '--reference_text', required=False, default=None,
                        help='path of .txt file of reference text')
    parser.add_argument('-s', '--segmentation_threshold', required=False,
                        default=None, type=int,
                        help='number of words or seconds of every text segment')
    parser.add_argument('-m', '--method_of_segmentation', required=False,
                        default=None,
                        help='Choice between "fixed_size_text" and '
                             '"fixed_window"')

    args = parser.parse_args()
    classifiers_attributes = load_classifiers(args.classifiers_path)
    features, feature_names, metadata = \
        get_asr_features(args.input, args.google_credentials,
                         classifiers_attributes, args.reference_text,
                         args.segmentation_threshold,
                         args.method_of_segmentation)
    print("Features names:\n {}".format(feature_names))
    print("Features:\n {}".format(features))
    print("Metadata:\n {}".format(metadata))

