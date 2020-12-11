import asr
import text_scoring as ts
import numpy as np
import fasttext


def load_reference_data(path):
    text = open(path).read()
    return text


def load_text_embedding_model(model_name="cc.en.300.bin"):
    # download https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
    model = fasttext.load_model(model_name)
    return model


def text_features(model, text):
    features = []
    features_names = []

    words = text.split(' ')
    features_t = []
    for w in words:
        features_t.append(model[w])
    features_t = np.array(features_t)
    features_m = np.mean(features_t, axis=0)

    for f in range(len(features_m)):
        features.append(features_m[f])
        features_names.append(f'fast_text_model_emeddings_{f}')

    # TODO: add pretrained model posteriors, e.g. P(y=negative|x) etc

    return features, features_names


def get_asr_features(input_file, embedding_model,
                     google_credentials, reference_text=None):
    """
    Extract text features from ASR results of a speech audio file
    :param input_file: path to the audio file
    :param google_credentials: path to the ASR google credentials file
    :param reference_text:  path to the reference text
    :return:
     - features: list of text features extracted
     - embedding_model: fasttext model varbiable
       (need to initiate with load_text_embeding_model)
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
                                                       data)

    features += features_text
    feature_names += features_names_text

    return features, feature_names, metadata

    
if __name__ == "__main__":
    get_asr_features()


