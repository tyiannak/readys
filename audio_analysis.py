import wave
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis.audioTrainTest import load_model, classifier_wrapper
from pyAudioAnalysis import MidTermFeatures as aF
import numpy as np
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioBasicIO as aio

def get_wav_properties(wav_path):
    """
    Reads sampling rate and duration of an WAV audio file
    :param wav_path: path to the WAV file
    :return: sampling rate in Hz, duration in seconds
    """
    with wave.open(wav_path, "rb") as wave_file:
        fs = wave_file.getframerate()
        duration = wave_file.getnframes() / float(fs)
    return fs, duration

def audio_based_feature_extraction(input_file):
    #silence features
    fs,dur = get_wav_properties(input_file)
    fs, x = aio.read_audio_file(input_file)
    seg_limits_short = aS.silence_removal(x, fs, 0.06, 0.06, 0.07)
    seg_limits_long = aS.silence_removal(x, fs, 0.1, 0.1, 0.3)

    #short windows
    total_speech_short = 0.0
    silence_durations_short = []
    counter = 0
    for k in seg_limits_short:
        if counter == 0:
            silence_durations_short.append(k[0])
        elif counter == (len(seg_limits_short) - 1):
            p = counter - 1
            silence_durations_short.append(k[0] - seg_limits_short[p][1])
            silence_durations_short.append(dur - k[1])
        else:
            p = counter - 1
            silence_durations_short.append(k[0] - seg_limits_short[p][1])
        word_speech_short = k[1] - k[0]
        total_speech_short = total_speech_short + word_speech_short
        counter = counter + 1
    speech_ratio_short = total_speech_short / dur
    speech_ratio_short = float("{:.2f}".format(speech_ratio_short))
    number_of_pauses_short = len(silence_durations_short)
    silence_durations_short = np.array(silence_durations_short)
    std_short = np.std(silence_durations_short)
    std_short = float("{:.2f}".format(std_short))
    average_silence_dur_short = np.mean(silence_durations_short)
    average_silence_dur_short = float("{:.2f}".format(average_silence_dur_short))
    silence_seg_per_minute_short = float("{:.2f}".format(number_of_pauses_short/(dur/60.0)))
    word_rate_in_speech_short = len(seg_limits_short)/total_speech_short
    word_rate_in_speech_short = float("{:.2f}".format(word_rate_in_speech_short))

    #long windows
    total_speech_long = 0.0
    counter = 0
    silence_durations_long = []
    for k in seg_limits_long:
        if counter == 0:
            silence_durations_long.append(k[0])
        elif counter == (len(seg_limits_long) - 1):
            p = counter - 1
            silence_durations_long.append(k[0] - seg_limits_long[p][1])
            silence_durations_long.append(dur - k[1])
        else:
            p = counter - 1
            silence_durations_long.append(k[0] - seg_limits_long[p][1])
        word_speech_long = k[1] - k[0]
        total_speech_long = total_speech_long + word_speech_long
        counter = counter + 1
    speech_ratio_long = total_speech_long / dur
    speech_ratio_long = float("{:.2f}".format(speech_ratio_long))
    number_of_pauses_long = len(silence_durations_long)
    silence_durations_long = np.array(silence_durations_long)
    std_long = np.std(silence_durations_long)
    std_long = float("{:.2f}".format(std_long))
    average_silence_dur_long = np.mean(silence_durations_long)
    average_silence_dur_long = float("{:.2f}".format(average_silence_dur_long))
    silence_seg_per_minute_long = float("{:.2f}".format(number_of_pauses_long / (dur / 60.0)))
    word_rate_in_speech_long = len(seg_limits_long) / total_speech_long
    word_rate_in_speech_long = float("{:.2f}".format(word_rate_in_speech_long))

    #classification features
    # Load classifier:

    classifier, mean, std, classes, mid_window, mid_step, short_window, \
    short_step, compute_beat = load_model("segment_classifier")

    # read audio file and convert to mono
    sampling_rate, signal = audioBasicIO.read_audio_file(input_file)
    signal = audioBasicIO.stereo_to_mono(signal)

    if sampling_rate == 0:
        # audio file IO problem
        return -1, -1, -1
    if signal.shape[0] / float(sampling_rate) < mid_window:
        mid_window = signal.shape[0] / float(sampling_rate)

    # feature extraction:
    mid_features, s, _ = \
        aF.mid_feature_extraction(signal, sampling_rate,
                                  mid_window * sampling_rate,
                                  mid_step * sampling_rate,
                                  round(sampling_rate * short_window),
                                  round(sampling_rate * short_step))
    classes = []

    # take every sample (every mid term window) and not every feature
    mid_features = mid_features.tolist()
    tlist = list(zip(*mid_features))
    tlist = np.array(tlist)
    for i in tlist:
        print(i)
        feature_vector = (i - mean) / std  # normalization
        print(feature_vector)
        class_id, probability = classifier_wrapper(classifier, "svm_rbf",
                                                   feature_vector)
        classes.append(class_id)
    num_of_highs = 0
    num_of_neutrals = 0
    num_of_lows = 0
    for i in classes:
        if i == 0.0:
            num_of_highs = num_of_highs + 1
        elif i == 1.0:
            num_of_neutrals = num_of_neutrals + 1
        else:
            num_of_lows = num_of_lows + 1
    high_percentage = float("{:.2f}".format(num_of_highs * 100 / len(classes)))
    neutral_percentage = float("{:.2f}".format(num_of_neutrals * 100 / len(classes)))
    low_percentage = float("{:.2f}".format(num_of_lows * 100 / len(classes)))

    #list of features and feature names
    feature_names = ["Average silence duration short (sec)","Average silence duration long (sec)","Silence segments per minute short (segments/min)","Silence segments per minute long (segments/min)","Std short","Std long","Speech ratio short (sec)","Speech ratio long (sec)","Word rate in speech short (words/sec)","Word rate in speech long (words/sec)","High class (%)","Neutral class (%)","Low class (%)"]
    features = [average_silence_dur_short,average_silence_dur_long,silence_seg_per_minute_short,silence_seg_per_minute_long,std_short,std_long,speech_ratio_short,speech_ratio_long,word_rate_in_speech_short,word_rate_in_speech_long,high_percentage,neutral_percentage,low_percentage]
    metadata = {
        "Number of pauses short": number_of_pauses_short,
        "Number of pauses long" : number_of_pauses_long,
        "Total speech duration short (sec)" : total_speech_short,
        "Total speech duration long (sec)"  : total_speech_long
    }
    return features,feature_names,metadata



# TODO Silence removal will go here
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