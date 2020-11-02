import asr
import text_scoring



def load_reference_data(path):
    text = open(path).read()
    return text


def text_based_feature_extraction(input_file,google_credentials,reference_text=None):

    feature_names =[]
    features = []
    asr_results, data,number_of_words,dur = asr.audio_to_asr_text(input_file,google_credentials)
    word_rate = float("{:.2f}".format(number_of_words / (dur / 60.0)))
    metadata = {"Asr timestamps" : asr_results,"Number of words" : number_of_words , "Total duration (sec)" : dur}
    if reference_text:
        ref_text = load_reference_data(reference_text)
        alignment, rec, pre = text_scoring.text_to_text_alignment_and_score(ref_text,
                                                                            data)
        if rec==0.0 or pre==0.0:
            f1=0.0
        else:
            f1=2*rec*pre/(rec+pre)
        rec =   float("{:.2f}".format(rec))
        pre = float("{:.2f}".format(pre))
        f1 = float("{:.2f}".format(f1))
        feature_names = ["Recall score (%)","Precision score(%)","F1 score (%)"]
        features = [rec,pre,f1]
        #temporal score calculation
        print(alignment)
        if alignment != []:
            adjusted_results = text_scoring.adjust_asr_results(asr_results,
                                                               alignment.second.elements,dur)
            print(adjusted_results)
            length = 0.5
            step = 0.1
            recall_list, precision_list, f1_list, Ref, Asr = text_scoring.windows(alignment.first.elements,
                                                                                  alignment.second.elements,
                                                                                  adjusted_results, length, step,dur)
            print(recall_list,precision_list,f1_list)
        else:
            length = 0.5
            step = 0.1
            i=length
            recall_list = []
            precision_list = []
            f1_list = []
            total_number_of_windows = 0
            while (i + length )< dur:
                total_number_of_windows += 1
                recall_list.append({"x": i, "y": 0})
                precision_list.append({"x": i, "y": 0})
                f1_list.append({"x": i, "y": 0})
                i += step
            Ref, Asr =["-"] * total_number_of_windows,["-"] *total_number_of_windows
        metadata["temporal_recall"] = recall_list
        metadata["temporal_precision"] = precision_list
        metadata["temporal_f1"] = f1_list
        metadata["temporal_ref"] = Ref
        metadata["temporal_asr"] = Asr

    feature_names.append("Word rate (words/min)")
    features.append(word_rate)


    return features,feature_names,metadata


    
if __name__ == "__main__":
    text_based_feature_extraction()


