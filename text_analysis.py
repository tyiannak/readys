import asr
import text_scoring as ts


def load_reference_data(path):
    text = open(path).read()
    return text


def text_based_feature_extraction(input_file,
                                  google_credentials,
                                  reference_text=None):

    feature_names =[]
    features = []
    asr_results, data, n_words, dur = asr.audio_to_asr_text(input_file,
                                                            google_credentials)
    word_rate = float("{:.2f}".format(n_words / (dur / 60.0)))
    metadata = {"asr timestamps": asr_results,
                "Number of words": n_words,
                "Total duration (sec)" : dur}
    if reference_text:
        ref_text = load_reference_data(reference_text)
        alignment, rec, pre = ts.text_to_text_alignment_and_score(ref_text,
                                                                  data)
        if rec == 0.0 or pre == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * rec * pre / (rec + pre)
        rec = float("{:.2f}".format(rec))
        pre = float("{:.2f}".format(pre))
        f1 = float("{:.2f}".format(f1))
        feature_names = ["Recall score (%)",
                         "Precision score(%)",
                         "F1 score (%)"]
        features = [rec, pre, f1]

        # temporal score calculation
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
            ref, asr_r = ["-"] * total_number_of_windows,["-"] * total_number_of_windows
        metadata["temporal_recall"] = recalls
        metadata["temporal_precision"] = precisions
        metadata["temporal_f1"] = f1s
        metadata["temporal_ref"] = ref
        metadata["temporal_asr"] = asr_r

    feature_names.append("Word rate (words/min)")
    features.append(word_rate)

    return features, feature_names, metadata


    
if __name__ == "__main__":
    text_based_feature_extraction()


