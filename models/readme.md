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
1. Open the config file and go under the 'text_classifier' field. Choose the type of classifier you wish to use (i.e. svm, xgboost, fasttext), by setting True only this specifc classifier.
2. In case of the fasttext classifier, you must place the path to your pretrained embeddings vector at the field pretrained_embedding_vectors.
3. In case of SVM, set its parameters.
4. If you wish to reduce your dataset for debbuging reasons, use the hop_samples variable.
5. Run the script:
	- case 1: Using the whole embedding model:
          ```python3 train_text.py -a "input" -p "wiki.en.bin" -o "output_classifier_dictionary_path"```
    - case 2: Using an embeddings limit in order to avoid memory errors:
          ```python3 train_text.py -a "input" -p "wiki.en.vec" -o "output_classifier_dictionary_path" -l embeddings_limit ```
   
   Where input is input data in csv format with one column "transcriptions" and one column "labels".
   The output classifier dictionary will be saved at the defined output_classifier_dictionary_path. 


## Test text
- case 1: Using the whole embedding model:
          ```python3 test_text.py -i "input_string" -p "wiki.en.bin"```
- case 2: Using an embeddings limit in order to avoid memory errors:
          ```python3 test_text.py -i "input_string" -p "wiki.en.vec" -l embeddings_limit ```
  
Where input_string is the input text that we want to classify , wiki.en.bin is the path of the pretrained fasttext model and classifier_dictionary_path is the path of the trained classifier. 


# Training audio classifiers 
## Download training data 
```
https://drive.google.com/drive/folders/1-vDw54Nh6rNtMzmKV5kzEvrpR75AYjbJ?usp=sharing 
```

## Train the audio classifiers 
1. Open the config file and go under the 'audio_classifier' field. Choose the type of classifier you wish to use (i.e. svm, xgboost), by setting True only this specifc classifier.
2. In case of SVM, set its parameters.
3. Set the parameters for the segment based feature extraction, using the field basic_features_params
4. Run ```python3 train_audio.py -i "input_path" -o "output_classifier_dictionary_path" ```

  Where input_path is the path of the directory which contains audio organized in folders of classes. The output classifier dictionary will be saved at the defined output_classifier_dictionary_path.


## Test audio 
Run: ```python3 test_audio.py -i "input_wav" -c "classifier_dictionary_path"```

Where classifier_dictionary_path is the path of the trained classifier and input.wav is the input audio file.
