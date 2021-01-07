from train_text import extract_fast_text_features
import argparse
import pickle as cPickle
import re
import pandas as pd
#segment-level classification of text to classify into aggregated classes
#input:
#       data: string(text) that we want to classify , fasttext_pretrained_model: for feature extraction (vectors) , svm_model: for final classification,
#             classes_file : the path of csv file that contains the classes that this model predicts
#output:
#       dictionary : a dictionary that has as elements the classes and as values the percentage (%) that this data belongs to each class
def predict_text_labels(data,fasttext_pretrained_model,svm_model,classes_file):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', data)
    num_of_samples = len(sentences)
    feature_matrix, _, _ = extract_fast_text_features(sentences,fasttext_pretrained_model)
    print(feature_matrix.shape)
    with open(svm_model, 'rb') as fid:
        classifier = cPickle.load(fid)
        mean = cPickle.load(fid)
        std = cPickle.load(fid)
    df = pd.read_csv(classes_file)
    classes = df['classes'].tolist()
    dictionary = {}
    for label in classes:
        dictionary[label] = 0
    for sample in feature_matrix: # for each sentence:
        class_id = classifier.predict((sample - mean) / std)
        label = class_id[0]
        dictionary[label] += 1
    for label in dictionary:
        # TODO: Replace this aggregation by posterior-based aggregation
        dictionary[label] = (dictionary[label] *100) / num_of_samples
    return dictionary



##the above statement is true when running from command line
##specify arguments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_data_path", help="the path of our data samples (csv with column transcriptions and one column labels)")
    parser.add_argument("fasttext_pretrained_model_path", help="the path of fasttext pretrained model (.vec file)")
    parser.add_argument("svm_model_path", help="the path of the classifier")
    parser.add_argument("file_of_classes_path" ,help = "the path of csv file that contains the name of classes of this specific model")
    args = parser.parse_args()
    predict_text_labels(args.input_data_path,args.fasttext_pretrained_model_path,args.svm_model_path,args.file_of_classes_path)

























