import json
import asr
import text_scoring
import pandas as pd


def load_conf_file(path):
    with open(path) as f:
        conf = json.load(f)
    return conf


def load_reference_data(path):
    text = open(path).read()
    return text


def main():
   
    conf = load_conf_file('config.json')

    asr_results, data = asr.audio_to_asr_text(conf['audiofile'],
                                              conf['google_credentials'])
    #print(asr_results)
    print(data)
    ref_text = load_reference_data(conf['reference_text'])
    alignment, rec, pre = text_scoring.text_to_text_alignment_and_score(ref_text,
                                                                        data)
    df = pd.DataFrame({"alignment":[alignment],"recall": [rec],"precision":[pre]})
    df.to_pickle("score.pkl")
    print('Recall score:', rec, '%')
    print('Precision score:', pre, '%')
    #print('Avarage score:',"%.1f" % (2 * rec * pre / (rec + pre)), '%')
    print ('Alignment:','\n', alignment)
   
    #print(asr_results)
    adjusted_results = text_scoring.adjust_asr_results(asr_results,
                                                       alignment.second.elements)
    #print(adjusted_results)
    #print(adjusted_results)
    length = 0.5
    step = 0.1
    recall_list,precision_list=text_scoring.windows(alignment.first.elements, alignment.second.elements,
                               adjusted_results, length, step)
    df2 = pd.DataFrame(data=recall_list)
    df3 = pd.DataFrame(data=precision_list)
    df2.to_pickle("recall_temporal.pkl")
    df3.to_pickle("precision_temporal.pkl")
  

    
if __name__ == "__main__":
    main()


