import Levenshtein
import json
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner
import asr
import text_scoring


def load_conf_file(path):
    with open(path, 'r') as f:
        conf = json.load(f)
    return conf


def load_reference_data(path):
    text = open(path).read()
    return text


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


def main():
    conf = load_conf_file('config.json')
    
    asr_results, data = asr.audio_to_asr_text(conf['audiofile'],
                                              conf['google_credentials'])
    print(asr_results)
    ref_text = load_reference_data('reference.txt')
    alignment, rec, pre = text_scoring.text_to_text_alignment_and_score(ref_text,
                                                                        data)
    print('Recall score:', rec, '%')
    print('Precision score:', pre, '%')
    print('Avarage score:',"%.1f" % (2 * rec * pre / (rec + pre)), '%')
    print ('Alignment:','\n',alignment)
    #print(alignment.first.elements)
    #print(alignment.second.elements)
    #print(asr_results)
    print(asr_results)
    adjusted_results = text_scoring.adjust_asr_results(asr_results,
                                                       alignment.second.elements)
    print(adjusted_results)
    #print(adjusted_results)
    length = 0.5
    step = 0.1
    text_scoring.windows(alignment.first.elements,
                         alignment.second.elements, adjusted_results,
                         length, step)


if __name__ == "__main__":
    main()


