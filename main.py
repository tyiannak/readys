import sys 
import numpy as np
#import pyaudio
import struct
import scipy.fftpack as scp
import matplotlib as tpl
import os
import io  
import json

from pydub import AudioSegment
from google.cloud.speech_v1 import enums
from google.cloud import speech
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner,StrictGlobalSequenceAligner, LocalSequenceAligner


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
	with open("asrPrediction.txt", "w") as f: 
		f.write(data) 
	return my_results,data;

def load_reference_data(path):
	text = open(path).read()
	return text

def to_translation_map(iterable):						#a function to make a set of junk (None) characters
    return {key: None for key in iterable}

#-----------------------------OLD WAY VIA DIFFLIB/IT WILL BE REPLACED--------------------------------------------------
def text_to_text_alignment_and_score(text_ref,text_pred):

	text_ref = text_ref.lower()    #convert texts to consist of lower case characters only
	text_pred = text_pred.lower()
	iterable = [".",","]
	translation_map = str.maketrans(to_translation_map(iterable)) #convert the reference text in order not to contain , and . (junk characters)
	text_ref = text_ref.translate(translation_map)

	# Create sequences to be aligned.
	a = Sequence(text_ref.split())
	b = Sequence(text_pred.split())
	#print('Sequence A:', a)
	#print('Sequence B:', b)
	
	# Create a vocabulary and encode the sequences.
	v = Vocabulary()
	aEncoded = v.encodeSequence(a)
	bEncoded = v.encodeSequence(b)
	#print('Encoded A:', aEncoded)
	#print('Encoded B:', bEncoded)

	# Create a scoring and align the sequences using global aligner.
	scoring = SimpleScoring(1, 0)
	aligner = GlobalSequenceAligner(scoring, 0)
	f,score, encodeds = aligner.align(aEncoded, bEncoded,text_ref.split(),text_pred.split(), backtrace=True)
	#print(f)
	#print(score)
	#print(encodeds)

	# Iterate over optimal alignments and print them.
	for encoded in encodeds:
	    alignment = v.decodeSequenceAlignment(encoded)
	    #print ('Alignment:','\n',alignment)
	    Score=alignment.score*100/len(text_ref.split())
	    #print ('Score:',Score ,'%')
	return alignment,Score
	    
	#return matches #first element is always from reference text and second from asr



def main():
	conf=load_conf_file('config.json')
	
	#asr_results,data=audio_to_asr_text(conf['audiofile'],conf['google_credentials'])

	ref_text=load_reference_data('reference.txt')
	data = open('asrPrediction.txt').read()
	alignment,score=text_to_text_alignment_and_score(ref_text,data)
	print ('Alignment:','\n',alignment)
	print ('Score:',score ,'%')

if __name__ == "__main__":
    main()

"""
#------------------------------OLD VERSION/TRASH-------------------------------------------
#--------------------------text only score 1/Similarity------------------------------------
#=========================first way===========================================
#via difflib,compare two documents and ratio() returns a float in [0, 1], measuring the similarity of the sequences
def to_translation_map(iterable):						#a function to make a set of junk (None) characters
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








