import pickle as cPickle
import re
import pandas as pd
import numpy as np
import argparse
import fasttext
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk import WordPunctTokenizer
import contractions
import num2words
from gensim.models import KeyedVectors
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from collections import Counter

nltk.download('punkt')
nltk.download('wordnet')
eps = np.finfo(float).eps
seed = 500


def load_dataset(data, class_file_name, hop_samples=None):
    # load our samples
    df = pd.read_csv(data)
    transcriptions = df['transcriptions'].tolist()
    labels = df['labels']

    if hop_samples:
        transcriptions = transcriptions[::hop_samples]
        labels = labels[::hop_samples]
    a = np.unique(labels)
    df = pd.DataFrame(columns=['classes'])
    df['classes'] = a
    df.to_csv(class_file_name, index=False)
    labels = labels.tolist()

    return transcriptions, labels


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


def check_balance(labels):

    class_samples = Counter(labels)
    balanced_degrees = 1/len(class_samples)
    imbalanced_degrees = {}
    for label in class_samples:
        imbalanced_degrees[label] = class_samples[label]/len(labels)
    print('--> Dataset samples per class: \n    {}'.format(imbalanced_degrees))

    is_imbalanced = False
    strategy = {}
    for label in imbalanced_degrees:
        if imbalanced_degrees[label] < 0.8 * balanced_degrees:
            is_imbalanced = True
            print("--> The class {} is undersampled.".format(label))

    return is_imbalanced


def text_preprocess(document):
    """
    Basic text preprocessing
    :param document: string containing input text
    :return: updated text
    """

    # Remove contractions
    #document = contractions.fix(document)

    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Numbers to words
    after_spliting = document.split()

    for index, word in enumerate(after_spliting):
        if word.isdigit():
            after_spliting[index] = num2words(word)
    document = ' '.join(after_spliting)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    lemmatizer = WordNetLemmatizer()

    tokens = document.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def standarization(features):
    """
    Normalize features (dimensions)
    :param features: unormalized features (num_of_samples x 300)
    :return: normalized_features , mean, std
    """
    X = np.array(features)
    mean = np.mean(X, axis=0) + eps
    std = np.std(X, axis=0) + eps
    standarized_features = np.array([(vec - mean) / std for vec in X])
    return standarized_features, mean, std


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
        X = np.array(features)
        mean = np.mean(X, axis=0) + eps
        std = np.std(X, axis=0) + eps
        # save one vector(300 dimensional) for every sample
        if i == 0:
            total_features = np.hstack((mean, std))
        else:
            total_features = np.vstack((total_features, np.hstack((mean, std))))
    return total_features


def train_svm(feature_matrix, labels, out_model):
    """
    Train svm classifier from features and labels (X and y)
    :param feature_matrix: np array (n samples x 300 dimensions) , labels:
    :param labels: list of labels of examples
    :param f_mean: mean feature vector (used for scaling)
    :param f_std: std feature vector (used for scaling)
    :param out_model: name of the svm model to save
    :return:
    """

    clf = svm.SVC(kernel="rbf", class_weight='balanced')
    svm_parameters = {'gamma': ['auto', 'scale'],
                      'C': [1e-1, 1, 5, 1e1]}

    scaler = StandardScaler()

    thresholder = VarianceThreshold(threshold=0)

    feature_selector = SelectKBest()
    features_num = feature_matrix.shape[1]
    k = [int(0.8 * features_num), int(0.9 * features_num), 'all']

    pca = PCA()
    n_components = [0.98, 0.99, 'mle', None]

    pipe = Pipeline(steps=[('scaler', scaler), ('thresholder', thresholder),
                           ('feature_selector', feature_selector),
                           ('pca', pca), ('SVM', clf)],
                    memory='sklearn_tmp_memory')

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)

    grid_clf = GridSearchCV(
        pipe, dict(feature_selector__k=k,
                   pca__n_components=n_components,
                   SVM__gamma=svm_parameters['gamma'],
                   SVM__C=svm_parameters['C']), cv=cv,
        scoring='f1_macro', n_jobs=-1)

    grid_clf.fit(feature_matrix, labels)

    clf_svc = grid_clf.best_estimator_
    clf_params = grid_clf.best_params_
    clf_score = grid_clf.best_score_
    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
    print("--> Parameters of best svm model: {}".format(clf_params))
    print("--> Best validation score:      {:0.5f} (+/-{:0.5f})".format(clf_score,
                                                                    clf_stdev))

    y_pred = clf_svc.predict(feature_matrix)

    print("--> Confusion Matrix of training + validation sets: \n {}".format(confusion_matrix(labels, y_pred)))

    with open(out_model, 'wb') as fid:
        cPickle.dump(clf_svc, fid)
    return clf_svc


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

    np.random.seed(seed)

    class_file_name = out_model + "_classenames.csv"

    print('--> Loading Dataset...')
    transcriptions, labels = load_dataset(data, class_file_name, 2)

    is_imbalanced = check_balance(labels)

    # extract features based on pretrained fasttext model
    print("--> Extracting text features")
    total_features = extract_fast_text_features(transcriptions, text_emb_model,
                                                embeddings_limit)

    x_train, x_test, y_train, y_test = train_test_split(total_features, labels,
                                                        test_size=0.15, random_state=seed,
                                                        stratify=labels)
    """
    if is_imbalanced:
        print('--> The dataset is imbalanced. Applying  Borderline-SMOTE SVM to balance the classes')
        #oversampler = SMOTE(random_state=seed, n_jobs=-1)
        #feature_matrix, labels = oversampler.fit_resample(feature_matrix, labels)
        #resampler = SMOTEENN(random_state=seed, n_jobs=-1)
        #feature_matrix, labels = resampler.fit_resample(feature_matrix, labels)
        resampler = SMOTETomek(random_state=seed, n_jobs=-1)
        #x_train_resambled, y_train_resambled = oversampler.fit_resample(x_train, y_train)
        x_train_resambled, y_train_resambled = resampler.fit_resample(x_train, y_train)
        _ = check_balance(y_train_resambled)
    """
    x_train_resambled, y_train_resambled = x_train, y_train
    # train svm classifier
    print("--> Training SVM classifier using GridSearchCV")
    clf = train_svm(x_train_resambled, y_train_resambled, out_model)

    # test svm
    y_pred = clf.predict(x_test)
    print('Classification report on test data:')
    print(classification_report(y_test, y_pred))

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
    print("--> Loading the text embeddings model")
    text_embeddings = load_text_embeddings(args.pretrained,
                                           args.embeddings_limit)
    fast_text_and_svm(args.annotation, text_embeddings,
                      args.outputmodelpath, args.embeddings_limit)
