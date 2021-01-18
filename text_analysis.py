import asr
import text_scoring as ts
import numpy as np
import fasttext
from gensim.models import KeyedVectors
import os
from models.test_text import predict_text_labels
import argparse
import re
from models.test_text import extract_features

def load_reference_data(path):
    text = open(path).read()
    return text


def load_text_embedding_model(text_embedding_path, embeddings_limit=None):
    """
    Loads the fasttext text representation model
    :param text_embedding_path: path to the fasttext .bin file
    :param embeddings_limit: limit of the number of embeddings.
        If None, then the whole set of embeddings is loaded.
    :return: fasttext model
    """
    # download https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
    if embeddings_limit:
        return KeyedVectors.load_word2vec_format(text_embedding_path,
                                                 limit=embeddings_limit)
    else:
        return fasttext.load_model(text_embedding_path)

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

def text_segmentation(text,segmentation_threshold=None,method=None,asr_timestamps=None):
    '''
    Break text into segments in accordance with a defined method
    :param text: the text to be segmented
    :param segmentation_threshold: the duration or magnitude of every segment (for example: 2sec window or 2 words per segment)
    :param method:
    -None: the text will be segmented into sentences based on the punctuation that asr has found
    -"fixed_size_text" : split text into fixed size segments (fixed number of words)
    -"fixed_window" : split text into fixed time windows (fixed seconds)
    :param asr_timestamps: the timestamps of words that asr has defined
    :return:
    -text_segmented : a list of segments of the text (every element of the list is a string)
    '''
    if not(method):
        text_segmented = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
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
            while word['st'] >= start_time_of_window and word['st'] <= (start_time_of_window + segmentation_threshold):
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

def text_features(model, text,models_directory,segmentation_threshold=None,method=None,asr_results=None,embeddings_limit=None):
    '''
    Features exported from models(classifiers)
    :param model: the fasttext pretrained model
    :param text: the text we want to extract features from (string)
    :param models_directory: the path of the directory which contains all text models (both models' file and .csv file of classes_names)
    :param segmentation_threshold: the duration or magnitude of every segment (for example: 2sec window or 2 words per segment)
    :param method:
    -None: the text will be segmented into sentences based on the punctuation that asr has found
    -"fixed_size_text" : split text into fixed size segments (fixed number of words)
    -"fixed_window" : split text into fixed time windows (fixed seconds)
    :param asr_timestamps: the timestamps of words that asr has defined
    :param embeddings_limit: embeddings_limit: limit of the number of embeddings.
        If None, then the whole set of embeddings is loaded.
    :return:
    - features: list of text features extracted
    - features_names: list of respective feature names
    '''
    features = []
    features_names = []

    '''
    words = text.split(' ')
    features_t = []
    for w in words:
        features_t.append(model[w])
    features_t = np.array(features_t)
    features_m = np.mean(features_t, axis=0)

    for f in range(len(features_m)):
        features.append(features_m[f])
        features_names.append(f'fast_text_model_emeddings_{f}')
    '''
    # TODO: load all segment-level models that have been trainied in
    #       a predefined path such as segment_models/text
    # TODO: add pretrained model posteriors, e.g. P(y=negative|x) etc
    dictionaries = []
    text_segmented = text_segmentation(text, segmentation_threshold, method, asr_results)
    print(text_segmented)
    feature_matrix , num_of_samples = extract_features(text_segmented,model,embeddings_limit)
    for filename in os.listdir(models_directory):
        if not (filename.endswith("_classesnames.csv")):
            model_path = os.path.join(models_directory, filename)
            classes_file_name = filename + "_classesnames.csv"
            classes_names_path = os.path.join(models_directory, classes_file_name)
            dictionary , _ = predict_text_labels(feature_matrix,num_of_samples,model_path, classes_names_path)
            dictionaries.append(dictionary)
    for dictionary in dictionaries:
        for label in dictionary:
            feature_string = label + "(%)"
            feature_value = dictionary[label]
            features_names.append(feature_string)
            features.append(feature_value)
    return features, features_names


def get_asr_features(input_file, google_credentials,
                     models_directory,embedding_model, reference_text=None,embeddings_limit=None,segmentation_threshold=None,method=None):
    """
    Extract text features from ASR results of a speech audio file
    :param input_file: path to the audio file
    :param google_credentials: path to the ASR google credentials file
    :models_directory: path of the directory which contains all trained text models (both models' file and .csv file of classes_names)
    :embedding_model: the pretrained fasttext model
    :param reference_text:  path to the reference text
    :embeddings_limit: limit of the number of embeddings.
        If None, then the whole set of embeddings is loaded.
    :param segmentation_threshold: the duration or magnitude of every segment (for example: 2sec window or 2 words per segment)
    :param method:
    -None: the text will be segmented into sentences based on the punctuation that asr has found
    -"fixed_size_text" : split text into fixed size segments (fixed number of words)
    -"fixed_window" : split text into fixed time windows (fixed seconds)
    :return:
     - features: list of text features extracted
     - feature_names: list of respective feature names
     - metadata: list of metadata
    """

    feature_names = []
    features = []
    # Step 1: speech recognition using google speech API:
    asr_results, data, n_words, dur = asr.audio_to_asr_text(input_file,
                                                            google_credentials)
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
            recalls, precisions, f1s, ref, asr_r = ts.windows(alignment.first.elements,
                                                              alignment.second.elements,
                                                              adjusted_results,
                                                              length,
                                                              step,dur)
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
    features_text, features_names_text = text_features(embedding_model,
                                                       data,
                                                       models_directory,
                                                       segmentation_threshold,
                                                       method,
                                                       asr_results,
                                                       embeddings_limit)

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
                        help="the directory which contains all trained classifiers (models' files + .csv classes_names files)")
    parser.add_argument("-p", "--pretrained_model_path",required=True,
                        help="the fast text pretrained model path")
    parser.add_argument('-r', '--reference_text', required=False, default=None,
                        help='path of .txt file of reference text')
    parser.add_argument('-l', '--embeddings_limit', required=False, default=None, type=int,
                        help='Strategy to apply in transfer learning: 0 or 1.')
    parser.add_argument('-s', '--segmentation_threshold', required=False, default=None, type=int,
                        help='number of words or seconds of every text segment')
    parser.add_argument('-m', '--method_of_segmentation', required=False, default=None,
                        help='Choice between "fixed_size_text" and "fixed_window"')
    args = parser.parse_args()
    embedding_model = load_text_embedding_model(args.pretrained_model_path,args.embeddings_limit)
    features,feature_names,metadata = get_asr_features(args.input, args.google_credentials,args.classifiers_path,
                                                          embedding_model,args.reference_text,args.embeddings_limit,
                                                          args.segmentation_threshold,args.method_of_segmentation)
    print("Features names:\n {}".format(feature_names))
    print("Features:\n {}".format(features))
    print("Metadata:\n {}".format(metadata))

