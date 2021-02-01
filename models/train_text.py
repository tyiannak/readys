import os
import re
import numpy as np
import argparse
import yaml
import fasttext
from gensim.models import KeyedVectors
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from feature_extraction import TextFeatureExtraction
from utils import load_dataset, check_balance, convert_to_fasttext_data, save_model

script_dir = os.path.dirname(__file__)
eps = np.finfo(float).eps
seed = 500
if not script_dir:
    with open(r'./config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
else:
    with open(script_dir + '/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)


def grid_init(clf, clf_name, parameters_dict):
    scaler = StandardScaler()

    thresholder = VarianceThreshold(threshold=0)

    pca = PCA()

    pipe = Pipeline(steps=[('scaler', scaler), ('thresholder', thresholder),
                           ('pca', pca), (clf_name, clf)],
                    memory='sklearn_tmp_memory')

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)

    grid_clf = GridSearchCV(
        pipe, parameters_dict, cv=cv,
        scoring='f1_macro', n_jobs=-1)

    return grid_clf


def train_basic_classifier(feature_matrix, labels):
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

    if config['text_classifier']['svm']:
        print("--> Training SVM classifier using GridSearchCV")
        clf = svm.SVC(kernel="rbf", class_weight='balanced')
        svm_parameters = {'gamma': ['auto', 'scale'],
                          'C': [1e-1, 1, 5, 1e1]}

        parameters_dict = dict(pca__n_components=n_components,
                               SVM__gamma=svm_parameters['gamma'],
                               SVM__C=svm_parameters['C'])

        grid_clf = grid_init(clf, "SVM", parameters_dict)

    elif config['text_classifier']['xgboost']:
        print("--> Training XGBOOST classifier using GridSearchCV")
        xgb = XGBClassifier(n_estimators=100)
        parameters_dict = dict(pca__n_components=n_components)
        grid_clf = grid_init(xgb, "XGBOOST", parameters_dict)

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


def basic_embedding_classifier(data, feature_extractor, out_model):
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
    transcriptions, labels, classnames = load_dataset(data, class_file_name, config['text_classifier']['hop_samples'])

    is_imbalanced = check_balance(labels)

    # extract features based on pretrained fasttext model

    total_features = feature_extractor.transform(transcriptions)

    if is_imbalanced:
        print('--> The dataset is imbalanced. Applying  SMOTETomek to balance the classes')
        resampler = SMOTETomek(random_state=seed, n_jobs=-1)
        x_train_resambled, y_train_resambled = resampler.fit_resample(total_features, labels)
        _ = check_balance(y_train_resambled)

        clf = train_basic_classifier(x_train_resambled, y_train_resambled)
    else:
        clf = train_basic_classifier(total_features, labels)

    model_dict = config
    model_dict['classifier'] = clf
    model_dict['classifier_classnames'] = classnames
    if out_model is None:
        save_model(model_dict, name="basic_classifier")
    else:
        save_model(model_dict, out_model=out_model)

    return None


def train_fastext_model(data, embeddings_limit, out_model):

    np.random.seed(seed)

    class_file_name = out_model + "_classenames.csv"

    print("--> Loading the text embeddings model")
    word_model = KeyedVectors.load_word2vec_format('wiki.en.vec',
                                                   limit=embeddings_limit)
    word_model.save_word2vec_format('my.vec')
    print('--> Loading Dataset...')
    transcriptions, labels, classnames = load_dataset(data, class_file_name, config['text_classifier']['hop_samples'])

    convert_to_fasttext_data(labels, transcriptions, 'train.txt')

    print("--> Training classifier using fasttext")
    model = fasttext.train_supervised(input='train.txt', epoch=25, lr=1.0,
                                      wordNgrams=2, verbose=2, minCount=1,
                                      loss="hs", dim=300, pretrainedVectors='my.vec', seed=seed)

    #model.save_model("fasttext_classifier.ftz")
    model_dict = config
    model_dict['classifier'] = clf
    model_dict['classifier_classnames'] = classnames

    if out_model is None:
        save_model(model_dict, name="FASTTEXT")
    else:
        save_model(model_dict, out_model=out_model)

    return None


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

    if config['text_classifier']['fasttext']:
        train_fastext_model(args.annotation, args.embeddings_limit, args.outputmodelpath)

    elif config['text_classifier']['svm'] or config['text_classifier']['xgboost']:
        feature_extractor = TextFeatureExtraction(args.pretrained,
                                                  args.embeddings_limit)
        basic_embedding_classifier(args.annotation, feature_extractor,
                          args.outputmodelpath)
    else:
        print('SVM and fasttext are the only supported classifiers.')
