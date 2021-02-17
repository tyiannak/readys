"""
Given an audio file this module is capable of :
 - extracting aggregates of audio features using
   models.test_audio.predict_audio_labels for all available segment models
 - extracting silence features
 - merging the above in a recording-level audio representation
"""

import wave
import numpy as np
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioBasicIO as aio
from models.test_audio import predict_audio_labels
import os
import argparse


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


def silence_features(segment_limits,dur):
    """
    Extract silence features based on audio segments
    :param segment_limits: A list of start-end timestamps for audio segments
    :param dur: The total duration of audio file
    :return: A list of silence_features, number_of_pauses, total_speech
    """
    total_speech = 0.0
    silence_durations = []
    counter = 0
    for k in segment_limits:
        if counter == 0:
            silence_durations.append(k[0])
        elif counter == (len(segment_limits) - 1):
            p = counter - 1
            silence_durations.append(k[0] - segment_limits[p][1])
            silence_durations.append(dur - k[1])
        else:
            p = counter - 1
            silence_durations.append(k[0] - segment_limits[p][1])
        word_speech = k[1] - k[0]
        total_speech = total_speech + word_speech
        counter = counter + 1
    speech_ratio = total_speech / dur
    speech_ratio = float("{:.2f}".format(speech_ratio))
    number_of_pauses = len(silence_durations)
    silence_durations = np.array(silence_durations)
    std = np.std(silence_durations)
    std = float("{:.2f}".format(std))
    average_silence_dur = np.mean(silence_durations)
    average_silence_dur = float("{:.2f}".format(average_silence_dur))
    silence_seg_per_minute = float("{:.2f}".format(number_of_pauses /
                                                   (dur / 60.0)))
    if total_speech == 0:
        word_rate_in_speech = 0
    else:
        word_rate_in_speech = len(segment_limits) / total_speech
    word_rate_in_speech = float("{:.2f}".format(word_rate_in_speech))
    silence_features = [average_silence_dur,
                        silence_seg_per_minute,
                        std,
                        speech_ratio,
                        word_rate_in_speech]
    return silence_features, number_of_pauses, total_speech


def audio_based_feature_extraction(input_file,models_directory):
    """
        Export all features for a wav file (silence based + classifiers based)
        :param input_file: the audio file
        :param models_directory: the directory which contains all trained
        classifiers (models' files + MEANS files)
        :return: features , feature_names , metadata
    """
    # A. silence features
    fs, dur = get_wav_properties(input_file)
    fs, x = aio.read_audio_file(input_file)
    print(input_file)
    print(len(x) / fs)
    # get the silence estimates using pyAudioAnalysis semisupervised approach
    # for different windows and steps
    if dur < 5:
        seg_limits_short = [[0, dur]]
        seg_limits_long = [[0, dur]]
    else:
        seg_limits_short = aS.silence_removal(x, fs, 0.5, 0.25, 0.5)
        seg_limits_long = aS.silence_removal(x, fs, 1.0, 0.25, 0.5)

    # short windows
    silence_features_short, number_of_pauses_short, total_speech_short = \
        silence_features(seg_limits_short, dur)
    # long windows
    silence_features_long, number_of_pauses_long, total_speech_long = \
        silence_features(seg_limits_long, dur)


    # B. segment model-based features
    # Load classifier:
    dictionaries = []
    for filename in os.listdir(models_directory):
        model_path = os.path.join(models_directory, filename)
        dictionary = predict_audio_labels(input_file, model_path)[0]
        dictionaries.append(dictionary)

    # list of features and feature names
    feature_names = ["Average silence duration short (sec)",
                     "Average silence duration long (sec)",
                     "Silence segments per minute short (segments/min)",
                     "Silence segments per minute long (segments/min)",
                     "Std short","Std long","Speech ratio short (sec)",
                     "Speech ratio long (sec)",
                     "Word rate in speech short (words/sec)",
                     "Word rate in speech long (words/sec)"]
    features = []

    for i in range(len(silence_features_short)):
        features.append(silence_features_short[i])
        features.append(silence_features_long[i])
    for dictionary in dictionaries:
        for label in dictionary:
            feature_string = label + "(%)"
            feature_value = dictionary[label]
            feature_names.append(feature_string)
            features.append(feature_value)

    metadata = {
        "Number of pauses short": number_of_pauses_short,
        "Number of pauses long" : number_of_pauses_long,
        "Total speech duration short (sec)" : total_speech_short,
        "Total speech duration long (sec)"  : total_speech_long
    }
    return features, feature_names, metadata



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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="the path of audio file" )
    parser.add_argument("-c", "--classifiers_path", required=True,
                        help="the directory which contains all "
                             "trained classifiers "
                             "(models' files + MEANS files)")
    args = parser.parse_args()
    features, feature_names,metadata = \
        audio_based_feature_extraction(args.input, args.classifiers_path)
    print("Features names:\n {}".format(feature_names))
    print("Features:\n {}".format(features))
    print("Metadata:\n {}".format(metadata))
