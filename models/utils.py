import re
import csv
import os
import pickle
import time
import yaml
import numpy as np
import pandas as pd
import num2words
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import svm
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from imblearn.combine import SMOTETomek


def text_preprocess(document):
    """
        Basic text preprocessing
        :param document: string containing input text
        :return: updated text
    """

    document.rstrip("\n")

    # Remove contractions
    # document = contractions.fix(document)

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
    preprocessed_text = document.lower()

    return preprocessed_text


def load_dataset(data, class_file_name, hop_samples=None):
    df = pd.read_csv(data)
    transcriptions = df['transcriptions'].tolist()

    labels = df['labels']

    if hop_samples:
        transcriptions = transcriptions[::hop_samples]
        labels = labels[::hop_samples]
    classnames = np.unique(labels)
    labels = labels.tolist()

    return transcriptions, labels, classnames


def folders_mapping(folders):
    """Return a mapping from folder to class and a mapping from class to folder."""
    folder2idx = {}
    idx2folder = {}
    for idx, folder in enumerate(folders):
        folder2idx[folder] = idx
        idx2folder[idx] = folder
    return folder2idx, idx2folder


def convert_to_fasttext_data(labels, transcriptions, filename):
    data = [label + " " + trans for label, trans in zip(labels, transcriptions)]
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, sep=' ',
              header=None, quoting=csv.QUOTE_NONE,
              quotechar="", escapechar=" ")


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


def split_data(x, y, test_size=0.2, fasttext_data=False, seed=None):

    if fasttext_data:
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=test_size, random_state=seed,
                                                            stratify=y)

        convert_to_fasttext_data(y_train, x_train, 'train.txt')
        convert_to_fasttext_data(y_test, x_test, 'test.txt')
        print("Splitted dataset into train (saved on train.txt) and test (saved on test.txt) subsets.")
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=test_size, random_state=seed,
                                                            stratify=y)
        return x_train, x_test, y_train, y_test


def save_model(model_dict, out_model=None, name=None, is_text=True):

    script_dir = os.path.dirname(__file__)
    if not script_dir:
        with open(r'./config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    else:
        with open(script_dir + '/config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    if is_text:
        out_folder = config["text_classifier"]["out_folder"]
    else:
        out_folder = config["audio_classifier"]["out_folder"]
    if out_model is None:
        timestamp = time.ctime()
        out_model = "{}_{}.pt".format(name, timestamp)
        out_model = out_model.replace(' ', '_')
    else:
        out_model = str(out_model)
        if '.pt' not in out_model or '.pkl' not in out_model:
            out_model = ''.join([out_model, '.pt'])

    if not script_dir:
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        out_path = os.path.join(out_folder, out_model)
    else:
        out_folder = os.path.join(script_dir, out_folder)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        out_path = os.path.join(out_folder, out_model)

    print(f"\nSaving model to: {out_path}\n")
    with open(out_path, 'wb') as handle:
        pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def grid_init(clf, clf_name, parameters_dict, is_imbalanced, seed=None):

    if is_imbalanced:
        print('--> The dataset is imbalanced. Applying  SMOTETomek to balance the classes')
        sampler = SMOTETomek(random_state=seed, n_jobs=-1)

    scaler = StandardScaler()

    thresholder = VarianceThreshold(threshold=0)

    pca = PCA()
    if is_imbalanced:
        pipe = Pipeline(steps=[('sampling', sampler), ('scaler', scaler), ('thresholder', thresholder),
                               ('pca', pca), (clf_name, clf)],
                        memory='sklearn_tmp_memory')

    else:
        pipe = Pipeline(steps=[('scaler', scaler), ('thresholder', thresholder),
                               ('pca', pca), (clf_name, clf)],
                        memory='sklearn_tmp_memory')

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)

    grid_clf = GridSearchCV(
        pipe, parameters_dict, cv=cv,
        scoring='f1_macro', n_jobs=-1)

    return grid_clf


def train_basic_segment_classifier(feature_matrix, labels, is_imbalanced, config, seed=None):
    """
    Train svm classifier from features and labels (X and y)
    :param feature_matrix: np array (n samples x 300 dimensions) , labels:
    :param labels: list of labels of examples
    :param f_mean: mean feature vector (used for scaling)
    :param f_std: std feature vector (used for scaling)
    :param out_model: name of the svm model to save
    :return:
    """

    n_components = [0.98, 0.99, 'mle', None]

    if config['svm']:
        print("--> Training SVM classifier using GridSearchCV")
        clf = svm.SVC(kernel="rbf", class_weight='balanced')
        svm_parameters = {'gamma': ['auto', 'scale'],
                          'C': [1e-1, 1, 5, 1e1]}

        parameters_dict = dict(pca__n_components=n_components,
                               SVM__gamma=svm_parameters['gamma'],
                               SVM__C=svm_parameters['C'])

        grid_clf = grid_init(clf, "SVM", parameters_dict, is_imbalanced)

    elif config['xgboost']:
        print("--> Training XGBOOST classifier using GridSearchCV")
        xgb = XGBClassifier(n_estimators=100)
        parameters_dict = dict(pca__n_components=n_components)
        grid_clf = grid_init(xgb, "XGBOOST", parameters_dict,
                             is_imbalanced, seed)

    else:
        print("The only supported basic classifiers are SVM and XGBOOST")
        return -1

    grid_clf.fit(feature_matrix, labels)

    clf_svc = grid_clf.best_estimator_
    clf_params = grid_clf.best_params_
    clf_score = grid_clf.best_score_
    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
    print("--> Parameters of best svm model: {}".format(clf_params))
    print("--> Best validation score:      {:0.5f} (+/-{:0.5f})".format(clf_score,
                                                                    clf_stdev))

    return clf_svc