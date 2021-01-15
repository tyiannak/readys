import asr
import text_scoring as ts
import numpy as np
import fasttext
from gensim.models import KeyedVectors
import os
from models.test_text import predict_text_labels
import argparse

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


def text_features(model, text,models_directory,embeddings_limit=None):
    '''
    Features exported from models(classifiers)
    :param model: the fasttext pretrained model
    :param text: the text we want to extract features from (string)
    :param models_directory: the path of the directory which contains all text models (both models' file and .csv file of classes_names)
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
    for filename in os.listdir(models_directory):
        if not (filename.endswith("_classesnames.csv")):
            model_path = os.path.join(models_directory, filename)
            classes_file_name = filename + "_classesnames.csv"
            classes_names_path = os.path.join(models_directory, classes_file_name)
            dictionary = predict_text_labels(text, model, model_path, classes_names_path, embeddings_limit)
            dictionaries.append(dictionary)
    for dictionary in dictionaries:
        for label in dictionary:
            feature_string = label + "(%)"
            feature_value = dictionary[label]
            features_names.append(feature_string)
            features.append(feature_value)
    return features, features_names


def get_asr_features(input_file, google_credentials,
                     models_directory,embedding_model, reference_text=None,embeddings_limit=None):
    """
    Extract text features from ASR results of a speech audio file
    :param input_file: path to the audio file
    :param google_credentials: path to the ASR google credentials file
    :models_directory: path of the directory which contains all trained text models (both models' file and .csv file of classes_names)
    :embedding_model: the pretrained fasttext model
    :param reference_text:  path to the reference text
    :embeddings_limit: limit of the number of embeddings.
        If None, then the whole set of embeddings is loaded.
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

    args = parser.parse_args()
    embedding_model = load_text_embedding_model(args.pretrained_model_path,args.embeddings_limit)
    features,feature_names,metadata = get_asr_features(args.input, args.google_credentials,args.classifiers_path,
                                                          embedding_model,args.reference_text,args.embeddings_limit)
    print("Features names:\n {}".format(feature_names))
    print("Features:\n {}".format(features))
    print("Metadata:\n {}".format(metadata))

