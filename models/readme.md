# Training text classifiers
## Download fasttext model
```
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
unzip wiki.en.zip
```
## Download training data 
```
https://drive.google.com/drive/folders/1FmesRxmxooICePJ-3gPQ6E9g7EQgkMpE?usp=sharing 
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
https://drive.google.com/drive/folders/1-vDw54Nh6rNtMzmKV5kzEvrpR75AYjbJ?usp=sharing 
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
