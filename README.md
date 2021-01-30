# readys
A Speech Analytics Python Tool for Speech Quality Assessment  

## Recording-level feature extraction
The goal of these modules is to extract features that provide an intermediate 
representation to speech recordings towards the assessment of speech quality. 

### Text: text_analysis.py 
In order to get text features from an audio file run the below command in your terminal 
```
python3 text_analysis.py -i wav_file -g google_credentials -c classifiers_path -p fasttext_pretrained_model -l embeddings_limit -s segmentation_threshold -m segmentation_method
```
Where: 

- wav_file : the path of audio file where the recording is stored

- google_credentials : a json file which contains the google credentials for 
  speech to text functionality 

- classifiers_path: the directory which contains all trained classifiers 
  (models' files + .csv classes_names files)

- fasttext_pretrained_model: the fast text pretrained model path 

- embeddings_limit(optional): if we don't want to load the whole pretrained 
  model then this parameter defines the number of vectors will load 

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
```
python3 audio_analysis.py -i wav_file -c classifiers_path
```
Where: 

- wav_file : the path of audio file 

- classifiers_path : the directory which contains all trained classifiers 
  (models' files + MEANS files)  
  
The feature_names , features and metadata will be printed

Note: See [models/readme](models/readme.md) for instructions how to train 
audio and text models 


# Training recording level classifiers 
## Download training data 
```
https://drive.google.com/drive/folders/1M0t5hj6PtbgEjzrgfcukeC5-MuZ541vl
```
## Download trained models for testing 
```
https://drive.google.com/drive/folders/1_aR6MCOmE5Q7KmBrqeB6Msjv0xsdFtc5
```
## Train the recording level classifiers 
```
python3 train_recording_level_classifier.py -i "inputs_path" -f "feature_type" -ct "classifier_type" -mn "model_name" -g "google_credentials" -a "audio_models_directory" -t "text_models_directory" -r "reference_text" -s "segmentation_threshold" -m "method_of_segmentation" -tp "train_percentage"
```
Where "inputs_path" is the path which containes samples devided into classes' directories, "feature_type" is "fused" "audio" or "text" for feature extraction, "classifier_type" is "svm" "svm_rbf" "knn" "gradientboosting" "ransomforest" or "extratrees" for the model that will be trained, "model_name" is the name of the model that will be trained,"google_credentials" is json file for google credentials, "audio_models_directory" is the path of audio models (both models + MEANS files), "text_models_directory" is the path for text models (both models + .csv files of classes' names) , "reference_text" (optional) is None for no reference text or the directory where reference texts (txt filed) are devided into class folders,"segmentation_threshold" (optional) is  the duration or magnitude of every segment (for example: 2sec window or 2 words per segment), "method_of_segmentation" (optional) None,fixed_size_text or fixed_window,"train_percentage" (optional) the percentage of the dataset that will be uses as train set (0.9 default).

## Test recording level
```
python3 test_recording_level.py -i "input_file" -mn "model_name" -mt "model_type" -g "google_credentials" -a "audio_models_directory" -t "text_models_directory"
```
Where "input_file" is the audio file to be classified, "model_name" the name of recording level classifier to be used, "model_type" the type of recording level classifier, "google_credentials" the json file of google credentials, "audio_models_directory" is the path of audio models (both models + MEANS files), "text_models_directory" is the path for text models (both models + .csv files of classes' names). 
