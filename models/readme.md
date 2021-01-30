# Training text classifiers
## Download fasttext model
```
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
unzip wiki.en.zip
```
## Download training data 
```
https://drive.google.com/drive/folders/1iE77qr5rNa2h9Tj_0J9rYJCph1KDiuEk
```
## Download trained models for testing 
```
https://drive.google.com/drive/folders/1_aR6MCOmE5Q7KmBrqeB6Msjv0xsdFtc5
```
## Train the text classifier
```
python3 train_text.py -a "input" -p "wiki.en.bin" -o "svm_output_classifier_path"
```
Where input is input data in csv format with one column "transcriptions" and one column "labels".
The output classifier will be saved at the defined "svm_output_classifier_path". 


## Test text 
```
python3 test_text.py -i "input_string" -p "wiki.en.bin" -c "svm_classifier_path" -n "classes_names" 
```
Where "input_string" is the input text that we want to classify , "wiki.en.bin" is the path of the pretrained fasttext model , "svm_classifier_path" is the path of the classifier we trained, "classes_names" is a csv file with the names of the classes(saved automatically when trained). 


# Training audio classifiers 
## Download training data 
```
https://drive.google.com/drive/folders/1iE77qr5rNa2h9Tj_0J9rYJCph1KDiuEk
```
## Download trained models for testing 
```
https://drive.google.com/drive/folders/1_aR6MCOmE5Q7KmBrqeB6Msjv0xsdFtc5
```
## Train the audio classifiers 
```
python3 train_audio.py -i "input_path" -o "svm_output_classifier_path" 
```
Where "input_path" is the path of the directory which contains audio organized in folders of classes, "svm_output_classifier_path" the path to which model will be saved.


## Test audio 
```
python3 test_audio.py -i "input_wav" -c "svm_classifier_path"
```

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
python3 train_recording_level_classifier.py -i "inputs_path" -f "feature_type" -g "google_credentials" -ct "classifier_type" -mn "model_name" -a "audio_models_directory" -t "text_models_directory" -emb "embedding_model_path" -l embeddings_limit
```
Where "inputs_path" is the path which containes samples devided into classes' directories, "feature_type" is "fused" "audio" or "text" for feature extraction,"google_credentials" is json file for google credentials, "classifier_type" is "svm" "svm_rbf" "knn" "gradientboosting" "ransomforest" or "extratrees" for the model that will be trained, "model_name" is the name of the model that will be trained, "audio_models_directory" is the path of audio models (both models + MEANS files), "text_models_directory" is the path for text models (both models + .csv files of classes' names) , "embedding_model_path" is the path of fasttext pretrained model, "embeddings_limit" is the limit of words that will be loaded from embedding model (None if the whole model is loaded).

## Test recording level
