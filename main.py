# command line 

import numpy as np
import pyaudio
import struct
import scipy.fftpack as scp
import termplotlib as tpl
import os
import io
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="asr-thodoris-c7dd00b4636e.json"
from google.cloud.speech_v1 import enums
from google.cloud import speech
language_code = "el-GR"
local_file_path = "temp.wav"

sample_rate_hertz = 8000
encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
client = speech.SpeechClient()
config = {
        "language_code": language_code,
        "sample_rate_hertz": sample_rate_hertz,
        "enable_word_time_offsets": True,
        "encoding": encoding,
    }
with io.open(local_file_path, "rb") as f:
    content = f.read()
audio = {"content": content}

#response = client.recognize(config=config, audio=audio)
my_results = []
response = client.long_running_recognize(config, audio).result()
for result in response.results:
    for alternative in result.alternatives:
        print('=' * 20)
        print('transcript: ' + alternative.transcript)
        print('confidence: ' + str(alternative.confidence))
        for w in alternative.words:
            my_results.append({"word": w.word, 
                "st": w.start_time.seconds + float(w.start_time.nanos) / 10**9,
                "et": w.end_time.seconds + float(w.end_time.nanos) / 10**9
                })
print(my_results)


