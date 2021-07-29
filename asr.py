import io
import os
from google.cloud.speech_v1.gapic import enums
from google.cloud import speech
from pyAudioAnalysis import audioBasicIO
from scipy.io import wavfile
import wave

MAX_FILE_DURATION = 30

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
    language_code = "en-US"

    fs, dur = get_wav_properties(audio_path)

    cur_pos = 0
    my_results = []
    data = ""

    number_of_words = 0

    # stereo to mono
    sampling_rate, signal = audioBasicIO.read_audio_file(audio_path)
    signal = audioBasicIO.stereo_to_mono(signal)
    wavfile.write(audio_path,fs,signal)
    #command1 = f"ffmpeg -i {audio_path} -ac 1 {audio_path} -y"
    #os.system(command1)

    while cur_pos < dur:

        encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
        client = speech.SpeechClient()
        config = {
                "language_code": language_code,
                "sample_rate_hertz": fs,
                "enable_word_time_offsets": True,
                "encoding": encoding,
            }
        cur_end = cur_pos + MAX_FILE_DURATION
        if dur < cur_end:
            cur_end = dur
            
        command = f"ffmpeg -i {audio_path} -ss {cur_pos} -to " \
                  f"{cur_end} temp.wav -loglevel panic -y"
        os.system(command)
        with io.open("temp.wav", "rb") as f:
            content = f.read()

       
        audio = {"content": content}

        response = client.long_running_recognize(config,audio).result()

        for flag, result in enumerate(response.results):
            alternative = result.alternatives[0]
            data += alternative.transcript
            number_of_words = number_of_words + len(alternative.words)
            for w in alternative.words:
                my_results.append({"word": w.word,
                                   "st": w.start_time.seconds +
                                         float(w.start_time.nanos) / 10**9 +
                                         cur_pos,
                                   "et": w.end_time.seconds +
                                         float(w.end_time.nanos) / 10**9 +
                                         cur_pos
                                   })

        cur_pos += MAX_FILE_DURATION
    return my_results, data,number_of_words,dur
