from gensim.models.keyedvectors import KeyedVectors
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pickle as cPickle
import re
import pandas as pd
import numpy as np
import argparse
import fasttext

def text_preprocess(document):
    """
    Basic text preprocessing
    :param document: string containing input text
    :return: updated text
    """
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))
    # Substitute multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Convert to lowercase
    document = document.lower()
    return document


def extract_fast_text_features(transcriptions, fasttext_pretrained_model):
    """
    For every sentence (example) extract 300 dimensional feature vector
    based on fasttext pretrained model

    :param transcriptions: list of samples-sentences ,
    :param fasttext_pretrained_model : path of fasttext pretrained
    model (.vec file)
    :return: fasttext_pretrained_model: numpy array (n x 300) -->
                                        n samples(sentences) x 300
                                        dimensions(features) normalized
    """

    # load first 500000 words from pretrained model
#    pretrained_model = \
#        KeyedVectors.load_word2vec_format(fasttext_pretrained_model,
#                                          limit=500)

    pretrained_model = fasttext.load_model(fasttext_pretrained_model)

    # for every sample-sentence
    total_features = []

    for i, k in enumerate(transcriptions):
        print(i)
        features = []
        k.rstrip("\n")
        # preprocessing
        pr = text_preprocess(k)
        # for every word in the sentence
        for word in pr.split():
            try:
                # find the most similar words in the dictionary,
                # if there is not any: continue with next word
#                result = pretrained_model.wv.most_similar(word)
                # take the first one of them (the one that matches the most)
#                most_similar_key, similarity = result[0]  # get the first match
                # take the vector of it
#                feature = pretrained_model[most_similar_key]
                feature = pretrained_model[word]
                # collect vectors of all words in one sample
                features.append(feature)
            except KeyError:
                continue

        # average the feature vectors for all the words in a sentence-sample
        X = np.asmatrix(features)
        mean = np.mean(X, axis=0) + 1e-14
        # save one vector(300 dimensional) for every sample
        if total_features == []:
            total_features = mean
        else:
            total_features = np.vstack((total_features, mean))

    # normalization of all features
    X = np.asmatrix(total_features)
    mean = np.mean(X, axis=0) + 1e-14
    std = np.std(X, axis=0) + 1e-14
    feature_matrix = np.array([])
    for i, f in enumerate(X):
        ft = (f - mean) / std
        if i == 0:
            feature_matrix = ft
        else:
            feature_matrix = np.vstack((feature_matrix, ft))
    return feature_matrix


def train_svm(feature_matrix,labels):
    """
    Train svm classifier from features and labels (X and y)
    :param feature_matrix: np array (n samples x 300 dimensions) , labels:
    :param labels: list of labels of examples
    :return:
    """
    print(feature_matrix.shape)
    parameters = {'kernel': ('poly', 'rbf'),
                  'C': [0.001, 0.01, 0.5, 1.0, 5.0, 10.0, 20.0]}
    svc = svm.SVC(gamma="scale")
    clf_svc = GridSearchCV(svc, parameters, cv=5)
    clf_svc.fit(feature_matrix, labels)
    print('Parameters of best svm model: {} \n'.format(clf_svc.best_params_))
    print('Mean cross-validated score of the '
          'best_estimator: {} \n'.format(clf_svc.best_score_))
    model_name = "svm_text_model"
    with open(model_name, 'wb') as fid:
        cPickle.dump(clf_svc, fid)
    return model_name


def fast_text_and_svm(myData, fasttext_pretrained_model):
    """

    :param myData: csv file with one column transcriptions (text samples)
                   and one column labels
    :param fasttext_pretrained_model: path of fast text pretrained model
                                      (.vec file)
    :return model_name: name of the svm model that saved ,
    :return class_file_name: name of csv file that contains the classes
                             of the model
    """
    # load first 500000 words from pretrained model
#    pretrained_model = \
#        KeyedVectors.load_word2vec_format(fasttext_pretrained_model,
#                                          limit=500000)

    #load our samples
    df = pd.read_csv(myData)
    transcriptions = df['transcriptions'].tolist()
    labels = df['labels']
    transcriptions = transcriptions[::10]
    labels = labels[::10]
    a = np.unique(labels)
    df = pd.DataFrame(columns=['classes'])
    df['classes'] = a
    class_file_name = 'svm_text_classes.csv'
    df.to_csv(class_file_name, index=False)
    labels = labels.tolist()

    #extract features based on pretrained fasttext model
    feature_matrix = extract_fast_text_features(transcriptions,
                                                fasttext_pretrained_model)

    #train svm classifier
    model_name = train_svm(feature_matrix, labels)
    print("Model saved with name:",model_name)
    print("Classes of this model saved with name:", class_file_name)
    return model_name, class_file_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--annotation",
                        help="the path of our data samples "
                             "(csv with column transcriptions "
                             "and one column labels)")
    parser.add_argument("-p", "--pretrained",
                        help="the path of fasttext pretrained model "
                             "(.vec file)")
    args = parser.parse_args()
    fast_text_and_svm(args.annotation, args.pretrained)
