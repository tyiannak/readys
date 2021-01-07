from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pickle as cPickle
import re
import pandas as pd
import numpy as np
import argparse
import fasttext
from sklearn.metrics import f1_score, make_scorer


def load_text_embeddings(text_embedding_path):
    """
    Loads the fasttext text representation model
    :param text_embedding_path: path to the fasttext .bin file
    :return: fasttext model
    """
    return fasttext.load_model(text_embedding_path)


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


def normalization(features):
    """
    Normalize features (dimensions)
    :param features: unnormalized features (num_of_samples x 300)
    :return: normalized_features , mean, std
    """
    X = np.asmatrix(features)
    mean = np.mean(X, axis=0) + 1e-14
    std = np.std(X, axis=0) + 1e-14
    normalized_features = np.array([])
    for i, f in enumerate(X):
        ft = (f - mean) / std
        if i == 0:
            normalized_features = ft
        else:
            normalized_features = np.vstack((normalized_features, ft))
    return normalized_features, mean, std


def extract_fast_text_features(transcriptions, text_emb_model):
    """
    For every sentence (example) extract 300 dimensional feature vector
    based on fasttext pretrained model

    :param transcriptions: list of samples-sentences ,
    :param text_emb_model : path of fasttext pretrained
    model (.vec file)
    :return: fasttext_pretrained_model: numpy array (n x 300) -->
                                        n samples(sentences) x 300
                                        dimensions(features) normalized
    """

    # for every sample-sentence
    total_features = []

    for i, k in enumerate(transcriptions):
        print(i)
        features = []
        k.rstrip("\n")
        # preprocessing
        pr = text_preprocess(k)
        for word in pr.split(): # for every word in the sentence
            feature = text_emb_model[word]
            #  feature = text_emb_model.get_word_vector(word)
            features.append(feature)

        # average the feature vectors for all the words in a sentence-sample
        X = np.asmatrix(features)
        mean = np.mean(X, axis=0) + 1e-14
        # save one vector(300 dimensional) for every sample
        if total_features == []:
            total_features = mean
        else:
            total_features = np.vstack((total_features, mean))
    return total_features


def train_svm(feature_matrix, labels, f_mean, f_std, out_model):
    """
    Train svm classifier from features and labels (X and y)
    :param feature_matrix: np array (n samples x 300 dimensions) , labels:
    :param labels: list of labels of examples
    :param f_mean: mean feature vector (used for scaling)
    :param f_std: std feature vector (used for scaling)
    :param out_model: name of the svm model to save
    :return:
    """
    parameters = {'kernel': ('poly', 'rbf'),
                  'C': [0.001, 0.01, 0.5, 1.0, 5.0]}
    svc = svm.SVC(gamma="scale")
    f1 = make_scorer(f1_score, average='macro')
    clf_svc = GridSearchCV(svc, parameters, cv=5, scoring=f1)
    clf_svc.fit(feature_matrix, labels)
    print('Parameters of best svm model: {} \n'.format(clf_svc.best_params_))
    print('Mean cross-validated score of the '
          'best_estimator: {} \n'.format(clf_svc.best_score_))
    with open(out_model, 'wb') as fid:
        cPickle.dump(clf_svc, fid)
        cPickle.dump(f_mean, fid)
        cPickle.dump(f_std, fid)
    return


def fast_text_and_svm(myData, text_emb_model, out_model):
    """

    :param myData: csv file with one column transcriptions (text samples)
                   and one column labels
    :param text_emb_model: text embeddings model (e.g. loaded using
                           load_text_embeddings() function
    :param out_model: name of the svm model to save
    :return model_name: name of the svm model that saved ,
    :return class_file_name: name of csv file that contains the classes
                             of the model
    """

    # load our samples
    df = pd.read_csv(myData)
    transcriptions = df['transcriptions'].tolist()
    labels = df['labels']

    # TODO: set that to 1
    hop_samples = 5
    transcriptions = transcriptions[::hop_samples]
    labels = labels[::hop_samples]
    a = np.unique(labels)
    df = pd.DataFrame(columns=['classes'])
    df['classes'] = a
    class_file_name = out_model + "_classenames.csv"
    df.to_csv(class_file_name, index=False)
    labels = labels.tolist()

    # extract features based on pretrained fasttext model
    total_features = extract_fast_text_features(transcriptions, text_emb_model)
    # normalization
    feature_matrix , mean , std = normalization(total_features)

    # train svm classifier
    train_svm(feature_matrix, labels, mean, std, out_model)
    print("Model saved with name:", out_model)
    print("Classes of this model saved with name:", class_file_name)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--annotation",
                        help="the path of our data samples "
                             "(csv with column transcriptions "
                             "and one column labels)")
    parser.add_argument("-p", "--pretrained",
                        help="the path of fasttext pretrained model "
                             "(.bin file)")
    parser.add_argument("-o", "--outputmodelpath",
                        help="path to the final svm model to be saved")

    args = parser.parse_args()

    text_embeddings = load_text_embeddings(args.pretrained)
    fast_text_and_svm(args.annotation, text_embeddings, args.outputmodelpath)
