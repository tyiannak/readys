import os
import io
from google.cloud.speech_v1 import enums
from google.cloud import speech
import audio_analysis


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
    sample_rate_hertz = audio_analysis.get_wav_sample_rate(audio_path)
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
    data = ""
    for flag, result in enumerate(response.results):
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
