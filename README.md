# readys
> Developed at [**MagCIL**](https://magcil.github.io), the Multimodal Analysis Group of the Institute of Informatics and Telecommunications, [NCSR "Demokritos"](https://www.demokritos.gr/), Athens, Greece.

A Speech Analytics Python Tool for Speech Quality Assessment  

## Recording-level feature extraction
The goal of these modules is to extract features that provide an intermediate 
representation to speech recordings towards the assessment of speech quality. 

### Text: text_analysis.py 
In order to get text features from an audio file run the below command in your terminal 
```
python3 text_analysis.py -i wav_file -g google_credentials -c classifiers_path -r reference_text -s segmentation_threshold -m segmentation_method
```
Where: 

- wav_file : the path of audio file where the recording is stored

- google_credentials : a json file which contains the google credentials for 
  speech to text functionality 

- classifiers_path: the directory which contains all text trained classifiers 

- reference_text(optional): path of .txt file of reference text 

- segmentation_threshold(optional): if you want to segment text by punctuation, 
  don't use this argument (or use None as value), 
  otherwise it is the number of words or seconds of every text segment 

- segmentation_method(optional): if the method of segmentation is punctuation 
  (by sentences) then don't use this argument (or use None as value), 
  otherwise use "fixed_size_text" for segmentation with fixedwords 
  per segment or "fixed_window" for segmentation with fixed time window. 

The feature_names , features and metadata will be printed. 

### Audio: audio_analysis.py 
In order to get audio features from audio file (silence features + 
classification features) run the below command in your terminal 
- Case 1: Not using pyaudio recording level features:
```
python3 audio_analysis.py -i wav_file -c classifiers_path
```
- Case 2: Adding pyaudio recording level features:
```
python3 audio_analysis.py -i wav_file -c classifiers_path -f
```
Where: 

- wav_file : the path of audio file 

- classifiers_path : the directory which contains all audio trained classifiers 
  
The feature_names , features and metadata will be printed

Note: See [models/readme](models/readme.md) for instructions how to train 
audio and text models

## About MagCIL

This repository comes from [MagCIL](https://magcil.github.io), the Multimodal Analysis Group of the
Institute of Informatics and Telecommunications at the National Centre for Scientific Research
"Demokritos" in Athens, Greece. We work on multimodal deep learning for audio, speech, music and
video, from fundamental research to deployed systems.

- Group website: <https://magcil.github.io>
- Publications: <https://magcil.github.io/pub.html>
- All our open-source code and data: <https://magcil.github.io/code.html>
- Work with us, in industry projects or research consortia: <https://magcil.github.io/collaborate.html>

Other tools from the group:

- [`pyAudioAnalysis`](https://github.com/tyiannak/pyAudioAnalysis) &mdash; audio feature extraction, classification, segmentation and applications
- [`deep_audio_features`](https://github.com/tyiannak/deep_audio_features) &mdash; training and using CNNs for audio classification (PyTorch)
- [`deepaudio-x`](https://github.com/magcil/deepaudio-x) &mdash; a PyTorch framework for audio classification
- [`deepaudio-lab`](https://github.com/magcil/deepaudio-lab) &mdash; a no-code platform for training, evaluating and deploying audio classifiers
- [`amvoc`](https://github.com/tyiannak/amvoc) &mdash; analysis of mouse vocal communication
- [`paura`](https://github.com/tyiannak/paura) &mdash; real-time audio recording and analysis

The full list is at <https://github.com/magcil>.
