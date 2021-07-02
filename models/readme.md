# 1. Segment - level classifiers

## 1.1 Text classifiers
### 1.1.1 Train
Download fasttext model:
```
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
unzip wiki.en.zip
```

Download training data, e.g. from:
`https://drive.google.com/drive/folders/1vgJf0hev60GeZbfyOiqU1YtX_h1FaGl8`

Train the text classifier:
1. Edit `config.yaml`. 
Choose classifier model (svm, xgboost, fasttext) and make it True.
2. In case of `fasttext`, place the path to the pretrained embeddings vector 
at `pretrained_embedding_vectors`
3. In case of SVM, set its parameters.
4. If you wish to reduce your dataset for debbuging reasons, use the 
`hop_samples` variable.
5. Run the training script script: 
    - case 1: Using the whole fasttext embedding model:
     
        ```python3 train_text.py --annotation valence_transcriptions_labels.csv --pretrained wiki.en.vec -o valence```
    - case 2: Using an embeddings limit in order to avoid memory errors:
         
        ```python3 train_text.py --annotation valence_transcriptions_labels.csv --pretrained wiki.en.vec -o valence_500K -l 500000``` 
    - case 3: Using bert embeddings: 
    	 
        ```python3 train_text.py --annotation valence_transcriptions_labels.csv --pretrained bert -o valence```
   
Input is training data in csv format with two columns: "transcriptions" 
(the text data examples) and "labels" (the ground truth target value of 
the classification task).
The output classifier dictionary will be saved at the defined 
`output_folder`. 

### 1.1.2 Test
```
python3 test_text.py -i "i am so sad please help me I need help this is so fucking awful. this is bad i am so angry you fucking idiot. Yes i am so happy this is really great" -c output_models/text/valence.pt
```

returns:

```
--> Loading the text embeddings model
--> Extracting text features
{'__label__val_high': 33.333333333333336, '__label__val_low': 66.66666666666667, '__label__val_neutral': 0.0}
```

## 1.2 Audio classifiers 
### 1.2.1 Train 
Download training data 
```
https://drive.google.com/drive/folders/12Ai25ZUCysAUUWRSiSd5WrsWRbB1gRuF
```

Train the audio classifiers 
1. Edit `config.yaml`, under the `audio_classifier` and choose classifier type
(i.e. svm, xgboost)
2. In case of SVM, set its parameters.
3. Set the parameters for the segment based feature extraction, 
using the field `basic_features_params`
4. Run the script: `python3 train_audio.py -i valence_train -o audio_valence`
 (input folder is supposed to have audio files as samples, organized in folders-classes). Model is saved as provided by the `-o` argument in the folder specified in the `audio_classifier.out_folder` field

### 1.2.2 Test 

```python3 test_audio.py -i test.wav -c output_models/audio/audio_arousal.pt```

returns

```
--> Extracting audio features
Predicted probability percentage for each class:
     ({'neutral': 0.0, 'strong': 80.0, 'weak': 20.0}, ['strong', 'strong', 'strong', 'strong', 'weak'])
```

# 2. Recording level classifiers 
## 2.1 Training

Download training data 
``` 
https://drive.google.com/drive/folders/1kIOdlztkGKfYZZONXYqT1h9OtJS9hFBP
``` 

To train:

1. Edit `config.yaml` and go under the `recording_level_classifier` field to 
   set `classifier_type` (svm, svm_rbf, knn, gradientboosting, extratrees, randomforest)
2. `raudio_num_features_discard`: the number of readys audio features to discard.
3. `pyaudio_num_features` : the number of pyaudio features to keep (use 'all' for all features otherwise use a number)
4. `gender` : 'male' or 'female' (this is not needed if the samples are already seperated)
5. In `features_type` choose the type of the segment-level features you wish to use 
(fused, text or audio) 
6. `audio_features` : 'late_fused' , 'fused' or 'audio' for late fusion of readys+pyaudio, early fusion of readys+pyaudio or readys_audio only, respectively
7. `late_fusion`: in the field `classifier_pyaudio` define the classifier's type of pyaudio features (lr,NaiveBayes or svm_rbf), in the fielad `classifier_raudio` define the classifier type of readys_audio features (lr,NaiveBayes or svm_rbf). 
8. In `metric` choose the validation metric (i.e. f1_macro) 
9. In `google_credentials` write the path of google credentials file
10. In `reference_text` choose False if no reference text is used or True if reference text is used (Attention!! If reference_text is True then the .txt reference    files should be located in the same directory as their relative wav files and have the same name)
11. In `text_segmentation_params` 
      - `segmentation_threshold` choose None if no threshold is used,otherwise choose an integer number which defines the number of words per segment or the               segment's duration (sec). 
      - `method_of_segmentation` choose None for text segmentation in sentences,"fixed_size_text" for segmentation in fixed number of words or "fixed_window" 
        for segmentation in fixed seconds.
7. In `audio_models_folder`
      `text_models_folder` and
      `out_folder` 
   write the paths of audio models,text models and recording level models respectively 
8. `audio_features`: if `fused` concatenates models' features with hand-crafted audio features from pyAudioAnalysis. 
9. `pyaudio_params`: if `audio_features` is fused, choose the hand-crafted feature parameters
10. Run training script e.g.  

```
python3 train_recording_level_classifier.py -i speeches -mn speeches
``` 

where the `-i` argument is the path to the folder that contains the folders 
(one folder for each class, each one containing the respective recordings - 
examples of each class). The model in the above example will be saved in 
`output_models/recording_level/speeches.pt`. 


## 2.2 Testing

Download trained models 
``` 
https://drive.google.com/drive/folders/1vuW93WZgb84Nt4iSjnuRTrEUfOUns4iV
``` 

(or use your own trained models)

Test recording-level:
1. Open the config file and define the google_credentials path and audio,
text models paths as before. 
2. Run `test_recording_level.py` e.g. `python3 test_recording_level.py -i test.wav -m output_models/recording_level/speeches.pt `
