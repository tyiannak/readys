# readys
A Speech Analytics Python Tool for Dyslexia Assessment  

## Get overall text features 
### text_analysis.py 
In order to get text features from audio file (handcrafted features + classification features) run the below command in your terminal 
```
python3 text_analysis.py -i wav_file -g google_credentials -c classifiers_path -p fasttext_pretrained_model -l embeddings_limit -s segmentation_threshold -m segmentation_method
```
Where: 

-wav_file : the path of audio file 

-google_credentials : a json file which containes the google credentials for speech to text functionality 

-classifiers_path : the directory which contains all trained classifiers (models' files + .csv classes_names files)

-fasttext_pretrained_model : the fast text pretrained model path 

-embeddings_limit(optional) : if we don't want to load the whole pretrained model then this parameter defines the number of vectors will load 

-segmentation_threshold(optional) : If you want to segment text by punctuation, don't use this argument (or use None as value), otherwise it is the number of words or seconds of every text segment 

-segmentation_method(optional) : If the method of segmentation is punctuation (by sentences) then don't use this argument (or use None as value) , otherwise use "fixed_size_text" for segmentation with fixedwords per segment or "fixed_window" for segmentation with fixed time window. 

The feature_names , features and metadata will be printed. 

## Get overall audio features
### audio_analysis.py 
In order to get audio features from audio file (silence features + classification features) run the below command in your terminal 
```
python3 audio_analysis.py -i wav_file -c classifiers_path
```
Where: 

-wav_file : the path of audio file 

-classifiers_path : the directory which contains all trained classifiers (models' files + MEANS files) 

The feature_names , features and metadata will be printed. 
