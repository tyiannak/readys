import sys 
import numpy as np
#import pyaudio
import struct
import scipy.fftpack as scp
import matplotlib as tpl
import os
import io  
import json
from difflib import SequenceMatcher
from difflib import get_close_matches
from pydub import AudioSegment
from google.cloud.speech_v1 import enums
from google.cloud import speech



def load_conf_file(path):
    with open(path, 'r') as f:
        conf = json.load(f)
    return conf

def audio_to_asr_text(audio_path,google_credentials_file):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=google_credentials_file
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
        flag = flag+1
        alternative = result.alternatives[0]
        if flag==1:
            data = alternative.transcript
        else:
            tran = alternative.transcript
            data = data +tran
        #print('=' * 20)
        #print('transcript: ' + alternative.transcript)
        #print('confidence: ' + str(alternative.confidence))
        for w in alternative.words:
            my_results.append({"word": w.word, 
               "st": w.start_time.seconds + float(w.start_time.nanos) / 10**9,
                "et": w.end_time.seconds + float(w.end_time.nanos) / 10**9
              })
    return my_results,data;

def load_reference_data(path):
    text = open(path).read()
    return text

def to_translation_map(iterable):                       #a function to make a set of junk (None) characters
    return {key: None for key in iterable}

#-----------------------------OLD WAY VIA DIFFLIB/IT WILL BE REPLACED--------------------------------------------------
def text_to_text_alignment(text_ref,text_pred):
    text_ref = text_ref.lower()    #convert texts to consist of lower case characters only
    text_pred = text_pred.lower()
    iterable = [".",","]
    translation_map = str.maketrans(to_translation_map(iterable)) #convert the reference text in order not to contain , and . (junk characters)
    text_ref = text_ref.translate(translation_map)
    list_asr = text_pred.split()   #split asr text into words
    list_reference = text_ref.split()  #split reference text into words
    i=0 #pointer for reference text 
    j=0 #pointer for asr text
    matches = []                    #words from reference text matched to words from asr text
    for word in list_reference:
        k=get_close_matches(word,[list_asr[j]],n=1,cutoff=0.5)  #cutoff=0.5 means that we match only if the word is similar at least 50% to the reference 
                                                                #n= the maximum number of closed matching words that we take
        #print(word,":",k)
        matches.append({word : k})
        if k!=[]:                       #if the reference word is omitted in the asr text, we don't increase pointer in asr text 
            j = j+1
        i=i+1
    return matches #first element is always from reference text and second from asr



def main():
    conf=load_conf_file('config.json')
    #print(conf['audiofile'])
    asr_results,data=audio_to_asr_text(conf['audiofile'],conf['google_credentials'])
    #print (asr_results)
    ref_text=load_reference_data('reference.txt')
    #matches=[]
    matches=text_to_text_alignment(ref_text,data)
    print(matches)

if __name__ == "__main__":
    main()

"""
#------------------------------OLD VERSION/TRASH-------------------------------------------
#--------------------------text only score 1/Similarity------------------------------------
#=========================first way===========================================
#via difflib,compare two documents and ratio() returns a float in [0, 1], measuring the similarity of the sequences
def to_translation_map(iterable):                       #a function to make a set of junk (None) characters
    return {key: None for key in iterable}

text1 = open('asrPrediction.txt').read() 
text2 = open('reference.txt').read()
text1 = text1.lower()    #convert texts to consist of lower case characters only
#print(text1)
text2 = text2.lower()
iterable = [".",","]
translation_map = str.maketrans(to_translation_map(iterable)) #convert the reference text in order not to contain , and . (junk characters)
text2 = text2.translate(translation_map)
#print(text2)
m = SequenceMatcher(None, text2, text1,False)
#print(m.ratio())
#k = m.ratio()
#print("Text only score:",k*100 , "%")


"""