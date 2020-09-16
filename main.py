import os
import io  
import Levenshtein
import json
from google.cloud.speech_v1 import enums
from google.cloud import speech
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner


def load_conf_file(path):
    with open(path, 'r') as f:
        conf = json.load(f)
    return conf


def audio_to_asr_text(audio_path, google_credentials_file):
    """
    Audio to asr using google speech API
    :param audio_path: wav audio file to analyze
    :param google_credentials_file:  path to google api credentials file
    :return:
        my_results: output dict of the format: ['word':..., 'st': ..., 'et':...]
        data: raw text output (not structured)
    """

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_file
    language_code = "el-GR"
    sample_rate_hertz = 44100
    encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
    client = speech.SpeechClient()
    config = {
            "language_code": language_code,
            "sample_rate_hertz": sample_rate_hertz,
            "enable_word_time_offsets": True,
            "encoding": encoding,
        }

    with io.open(audio_path, "rb") as f:
        content = f.read()
    audio = {"content": content}

    my_results = []
    response = client.long_running_recognize(config, audio).result()
    flag = 0
    for result in response.results:
        flag += 1
        alternative = result.alternatives[0]
        if flag == 1:
            data = alternative.transcript
        else:
            tran = alternative.transcript
            data += tran
        for w in alternative.words:
            my_results.append({"word": w.word, 
                               "st": w.start_time.seconds + 
                                     float(w.start_time.nanos) / 10**9,
                               "et": w.end_time.seconds + 
                                     float(w.end_time.nanos) / 10**9
                               })
    with open("asrPrediction.txt", "w") as f: 
        f.write(data)

    return my_results, data


def load_reference_data(path):
    text = open(path).read()
    return text


def to_translation_map(iterable): 
    #a function to make a set of junk (None) characters
    return {key: None for key in iterable}


def text_to_text_alignment_and_score(text_ref, text_pred):
    """
    Find a word to word alignment between two texts, considering the first is 
    the reference and the second the predicted
    :param text_ref: text reference
    :param text_pred: predicted text
    :return: 
    """

    text_ref = text_ref.lower() 
    text_pred = text_pred.lower()
    iterable = [".", ","]
    # convert the reference text in order not to contain , and . (junk characters)
    translation_map = str.maketrans(to_translation_map(iterable))
    text_ref = text_ref.translate(translation_map)

    # Create sequences to be aligned.
    a = Sequence(text_ref.split())
    b = Sequence(text_pred.split())
    
    # Create a vocabulary and encode the sequences.
    v = Vocabulary()
    a_enc = v.encodeSequence(a)
    b_enc = v.encodeSequence(b)

    # Create a scoring and align the sequences using global aligner.
    scoring = SimpleScoring(1, 0)
    aligner = GlobalSequenceAligner(scoring, 0)
    f, score, encodeds = aligner.align(a_enc, b_enc, text_ref.split(),
                                       text_pred.split(), backtrace=True)
    print(encodeds)

    # Iterate over optimal alignments and print them.
    for encoded in encodeds:
        alignment = v.decodeSequenceAlignment(encoded)
        rec = alignment.score * 100 / len(text_ref.split())
        pre = alignment.score * 100 / len(text_pred)
    return alignment, rec, pre
        

"""
def silence_removal(audio_path):
    audio, sample_rate = read_wave(audio_path)
    vad = webrtcvad.Vad(int(1))
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)

    # Segmenting the Voice audio and save it in list as bytes
    concataudio = [segment for segment in segments]
    joinedaudio = b"".join(concataudio)
    write_wave("Non-Silenced-Audio.wav", joinedaudio, sample_rate)
"""

def adjust_asr_results(asr_results, second):
    adjusted_results = []
    i = 0
    for j in range(0, len(second)):
        if asr_results[i]['word'].lower() == second[j]:
            adjusted_results.append(asr_results[i]) 
            i += 1
        else:
            if adjusted_results[j-1]['word']=='-':
                adjusted_results.append(adjusted_results[j-1])
            else:
                k = adjusted_results[j-1]['et']
                l = asr_results[i]['st']
                mean = (k+l) / 2
                adjusted_results.append({"word": second[j], "st": mean,
                                         "et" : mean})
    return adjusted_results


def calculate_score_after_alignment(A, B):
    total_score = 0.0
    k = len(A)
    ref_words = 0
    asr_words = 0
    for j in range(0, k):
        if A[j] != '-':
            ref_words += 1
        if B[j] != '-':
            asr_words += 1
        if A[j] == B[j]:
            total_score += 1.0
        elif B[j] != '-' and A[j] != '-':
            total_score = total_score + Levenshtein.ratio(A[j], B[j]) 
    rec = total_score * 100 / ref_words
    pre = total_score * 100 / asr_words
    return rec, pre


def windows(first, second, adjusted_results, length, step):
    if step == 0:
        raise ValueError("Parameter 'm' can't be 0")
    i = length  #center of window
    k = len(second)-1
    while i + length < adjusted_results[k]['et']:
        list_a=[]
        list_b=[]
        for j in range(0, len(second)):
            bottom = i - length
            up = i + length
            if adjusted_results[j]['st'] >= bottom and \
                    adjusted_results[j]['st'] <= up:
                list_a.append(first[j])
                list_b.append(second[j])
        rec, pre = calculate_score_after_alignment(list_a, list_b)
        print('Recall score from', "%.1f" % (i-length),
              'sec', 'to', "%.1f" %(i+length), 'sec', 'is:', rec, '%')
        print('Precision score from', "%.1f" % (i-length), 'sec', 'to', "%.1f"
              % (i+length), 'sec', 'is:', pre, '%')
        print('Avarage score:', "%.1f" % (2*rec*pre/(rec+pre)), '%')
        i += step


def main():
    conf = load_conf_file('config.json')
    
    asr_results, data = audio_to_asr_text(conf['audiofile'],
                                          conf['google_credentials'])
    print(asr_results)
    ref_text = load_reference_data('reference.txt')
    alignment, rec, pre = text_to_text_alignment_and_score(ref_text, data)
    print('Recall score:', rec, '%')
    print('Precision score:', pre, '%')
    print('Avarage score:',"%.1f" % (2 * rec * pre / (rec + pre)), '%')
    print ('Alignment:','\n',alignment)
    #print(alignment.first.elements)
    #print(alignment.second.elements)
    #print(asr_results)
    adjusted_results = adjust_asr_results(asr_results, alignment.second.elements)
    #print(adjusted_results)
    length = 0.3
    step = 0.3
    windows(alignment.first.elements, alignment.second.elements,
            adjusted_results, length, step)


if __name__ == "__main__":
    main()


