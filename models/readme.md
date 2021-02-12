# Training and testing segment - level classifiers

## Text classifiers

Download fasttext model:
```
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
unzip wiki.en.zip
```

Download training data, e.g. from:
```
https://drive.google.com/drive/folders/1FmesRxmxooICePJ-3gPQ6E9g7EQgkMpE?usp=sharing 
```

Train the text classifier:
1. Edit `config.yaml`, under the `text_classifier` field. Choose the type of classifier (svm, xgboost, fasttext), by setting True the respective field. 
2. In case of `fasttext`, place the path to the pretrained embeddings vector at the field `pretrained_embedding_vectors`.
3. In case of SVM, set its parameters.
4. If you wish to reduce your dataset for debbuging reasons, use the `hop_samples` variable.
5. Run the script: 
	- case 1: Using the whole embedding model:
        ```python3 train_text.py --annotation valence_transcriptions_labels.csv --pretrained wiki.en.vec -o valence```
    - case 2: Using an embeddings limit in order to avoid memory errors:
           ```python3 train_text.py --annotation valence_transcriptions_labels.csv --pretrained wiki.en.vec -o valence_500K -l 500000```
   
   Where input is input data in csv format with one column "transcriptions" and one column "labels".
   The output classifier dictionary will be saved at the defined output_classifier_dictionary_path. 


Test the text classifier:
```
python3 test_text.py -i "i am so sad please help me I need help this is so fucking awful. this is bad i am so angry you fucking idiot. Yes i am so happy this is really great" -c output_models/text/valence.pt
```
  

## Audio classifiers 
Download training data 
```
https://drive.google.com/drive/folders/1-vDw54Nh6rNtMzmKV5kzEvrpR75AYjbJ?usp=sharing 
```

Train the audio classifiers 
1. Edit `config.yaml`, under the `audio_classifier` field choose the type 
of classifier to use (i.e. svm, xgboost), by setting True the respective field
2. In case of SVM, set its parameters.
3. Set the parameters for the segment based feature extraction, using the field `basic_features_params`
4. Run, e.g. ```python3 train_audio.py -i valence_train -o audio_valence``` (input folder is supposed to have audio files as samples, organized in folders-classes). Model is saved as provided by the `-o` argument in the folder specified in the `audio_classifier.out_folder` field

Test audio classifier:
Run: ```python3 test_audio.py -i test.wav -c output_models/audio/audio_arousal.pt```


# Training recording level classifiers 
## Download training data 
``` 
https://drive.google.com/drive/folders/1kIOdlztkGKfYZZONXYqT1h9OtJS9hFBP
``` 
## Train the recording level classifiers 
1. Open the config file and go under the 'recording_level_classifier' field. 
   In "classifier_type" choose the type of classifier you wish to use (svm, svm_rbf,knn,gradientboosting,extratrees,randomforest)
2. In "features_type" choose the type of the features you wish to use (fused,text or audio) 
3. In "metric" choose the validation metric (i.e. f1_macro) 
4. In "google_credentials" write the path of google credentials file
5. In "reference_text" choose False if no reference text is used or True if reference text is used (Attention!! If reference_text is True then the .txt reference    files should be located in the same directory as their relative wav files and have the same name)
6. In "text_segmentation_params" 
      - 'segmentation_threshold' choose None if no threshold is used,otherwise choose an integer number which defines the number of words per segment or the               segment's duration (sec). 
      - 'method_of_segmentation' choose None for text segmentation in sentences,"fixed_size_text" for segmentation in fixed number of words or "fixed_window" 
        for segmentation in fixed seconds.
7. In "audio_models_folder"
      "text_models_folder" and
      "out_folder" 
   write the paths of audio models,text models and recording level models respectively. 
8. Run ```python3 models/train_recording_level_classifier.py -i "input_path" -mn "model_name"``` 

Where input_path is the directory where samples are devided into class folders and model_name is the name of the model that we are gonna train. 

## Testing recording level 
## Download trained models 
``` 
https://drive.google.com/drive/folders/1vuW93WZgb84Nt4iSjnuRTrEUfOUns4iV
``` 
## Test recording level 
1. Open the config file and define the google_credentials path and audio,text models paths as before. 
2. Run ```python3 models/test_recording_level.py -i "input" -m "model_path" ``` 

Where input is the input wav file and model_path is the path of the recording level model we are gonna use fore testing.
