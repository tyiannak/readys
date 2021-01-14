import pickle as cPickle
import re
import pandas as pd
import numpy as np
import argparse
import fasttext
from gensim.models import KeyedVectors
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
eps = np.finfo(float).eps


def load_text_embeddings(text_embedding_path, embeddings_limit=None):
    """
    Loads the fasttext text representation model
    :param text_embedding_path: path to the fasttext .bin file
    :param embeddings_limit: limit of the number of embeddings.
        If None, then the whole set of embeddings is loaded.
    :return: fasttext model
    """
    if embeddings_limit:
        return KeyedVectors.load_word2vec_format(text_embedding_path,
                                                 limit=embeddings_limit)
    else:
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
    :param features: unormalized features (num_of_samples x 300)
    :return: normalized_features , mean, std
    """
    X = np.asmatrix(features)
    mean = np.mean(X, axis=0) + eps
    std = np.std(X, axis=0) + eps
    normalized_features = np.array([])
    for i, f in enumerate(X):
        ft = (f - mean) / std
        if i == 0:
            normalized_features = ft
        else:
            normalized_features = np.vstack((normalized_features, ft))
    return normalized_features, mean, std


def extract_fast_text_features(transcriptions, text_emb_model,
                               embeddings_limit=None):
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

    for i, k in enumerate(transcriptions):
        features = []
        k.rstrip("\n")
        # preprocessing
        pr = text_preprocess(k)
        for word in pr.split(): # for every word in the sentence
            if embeddings_limit:
                try:
                    result = text_emb_model.similar_by_word(word)
                    most_similar_key, similarity = result[0]
                    feature = text_emb_model[most_similar_key]
                    features.append(feature)
                except:
                    continue
            else:
                feature = text_emb_model[word]
                features.append(feature)
        # average the feature vectors for all the words in a sentence-sample
        X = np.asmatrix(features)
        mean = np.mean(X, axis=0) + eps
        # save one vector(300 dimensional) for every sample
        if i == 0:
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
    clf = svm.SVC(kernel='rbf', class_weight='balanced')
    svm_parameters = {'gamma': ['auto', 'scale', 1e-3, 1e-4, 1e-5, 1e-6],
                      'C': [1e-3, 1e-2, 1e-1, 1, 0.5e1, 1e1]}

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)

    grid_clf = GridSearchCV(
        clf, dict(gamma=svm_parameters['gamma'],
                    C=svm_parameters['C']), cv=cv,
                   scoring='f1_macro', n_jobs=-1)

    grid_clf.fit(feature_matrix, labels)

    clf_svc = grid_clf.best_estimator_
    clf_params = grid_clf.best_params_
    clf_score = grid_clf.best_score_
    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]

    print("Parameters of best svm model: {}".format(clf_params))
    print("Best validation score:      {:0.5f} (+/-{:0.5f})".format(clf_score,
                                                                    clf_stdev))

    y_pred = clf_svc.predict(feature_matrix)

    print("Confusion Matrix: \n {}".format(confusion_matrix(labels, y_pred)))

    with open(out_model, 'wb') as fid:
        cPickle.dump(clf_svc, fid)
        cPickle.dump(f_mean, fid)
        cPickle.dump(f_std, fid)
    return


def fast_text_and_svm(data, text_emb_model, out_model,
                      embeddings_limit=None):
    """

    :param data: csv file with one column transcriptions (text samples)
                   and one column labels
    :param text_emb_model: text embeddings model (e.g. loaded using
                           load_text_embeddings() function
    :param out_model: name of the svm model to save
    :return model_name: name of the svm model that saved ,
    :return class_file_name: name of csv file that contains the classes
                             of the model
    """

    # load our samples
    df = pd.read_csv(data)
    transcriptions = df['transcriptions'].tolist()
    labels = df['labels']

    # TODO: set that to 1
    hop_samples = 2
    transcriptions = transcriptions[::hop_samples]
    labels = labels[::hop_samples]
    a = np.unique(labels)
    df = pd.DataFrame(columns=['classes'])
    df['classes'] = a
    class_file_name = out_model + "_classenames.csv"
    df.to_csv(class_file_name, index=False)
    labels = labels.tolist()

    # extract features based on pretrained fasttext model
    print("Extracting text features...")
    total_features = extract_fast_text_features(transcriptions, text_emb_model,
                                                embeddings_limit)
    # normalization
    print("Normalizing text features...")
    feature_matrix, mean, std = normalization(total_features)

    # train svm classifier
    print("Training SVM classifier using GridSearchCV...")
    train_svm(feature_matrix, labels, mean, std, out_model)
    print("Model saved with name:", out_model)
    print("Classes of this model saved with name:", class_file_name)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--annotation", required=True,
                        help="the path of our data samples "
                             "(csv with column transcriptions "
                             "and one column labels)")
    parser.add_argument("-p", "--pretrained", required=True,
                        help="the path of fasttext pretrained model "
                             "(.bin file)")
    parser.add_argument("-o", "--outputmodelpath", required=False,
                        default="SVM",
                        help="path to the final svm model to be saved")
    parser.add_argument('-l', '--embeddings_limit', required=False,
                        default=None, type=int,
                        help='Strategy to apply in transfer learning: 0 or 1.')

    args = parser.parse_args()
    text_embeddings = load_text_embeddings(args.pretrained,
                                           args.embeddings_limit)
    fast_text_and_svm(args.annotation, text_embeddings,
                      args.outputmodelpath, args.embeddings_limit)
