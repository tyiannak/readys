import re
import csv
import os
import pickle5 as pickle
import time
import numpy as np
import pandas as pd
from transformers import BertModel
from num2words import num2words
from collections import Counter
from collections import defaultdict
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit, LeaveOneGroupOut
from sklearn import preprocessing
from sklearn import svm
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier,\
    GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
import fasttext
from gensim.models import KeyedVectors
from pathlib import Path
import torch
import plotly
import plotly.subplots as plotly_sub
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioTrainTest as aT

y_true = []
y_pred = []

def text_preprocess(document):
    """
        Text preprocessing
        :param document: string containing input text
        :return: updated text
    """

    document.rstrip("\n")

    # Remove contractions
    # document = contractions.fix(document)

    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # Remove single characters from the start
    # document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Remove leading and trailing whitespaces 
    document.strip()
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    #document = re.sub(r'^b\s+', '', document)

    # Numbers to words
    after_spliting = document.split()

    for index, word in enumerate(after_spliting):
        if word.isdigit():
            after_spliting[index] = num2words(int(word))
    document = ' '.join(after_spliting)

    # Converting to Lowercase
    preprocessed_text = document.lower()

    return preprocessed_text


def load_text_embeddings(word_model_path,embeddings_limit=None):
    '''
    Loading the embedding
    :param word_model_path: the embedding path
    :param embeddings_limit: the embedding limit
    :return: the embedding model loaded
    '''
    print("--> Loading the fasttext embeddings")
    if embeddings_limit:
        word_model = KeyedVectors.load_word2vec_format(
            word_model_path, limit=embeddings_limit)
    else:
        word_model = fasttext.load_model(os.path.join(Path(__file__).parent,
                                                      word_model_path))
    return word_model


def load_text_classifier_attributes(classifier_path):
    '''
    Load the attributes of the text classifier that are saved into the classifier
    :param classifier_path: the path of the classifier
    :return: classifier,classes,pretrained_path,embeddings_limit,
    fasttext_model_path
    '''
    model_dict = pickle.load(open(classifier_path, 'rb'))
    if model_dict['embedding_model'] == 'bert':
        fasttext_model_path = None
        pretrained_path = model_dict['embedding_model']
        embeddings_limit = None
        print("--> Loading the bert embeddings")
        pretrained = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
        classifier = model_dict['classifier']
        max_len = model_dict['max_len']
    elif model_dict['classifier_type'] == 'fasttext':
        fasttext_model_path = model_dict['fasttext_model']
        print("--> Loading the fasttext model")
        classifier = fasttext.load_model(fasttext_model_path)
        embeddings_limit = None
        pretrained = None
        pretrained_path = None
        max_len = None
    else:
        fasttext_model_path = None
        pretrained_path = model_dict['embedding_model']
        embeddings_limit = model_dict['embeddings_limit']
        pretrained = load_text_embeddings(pretrained_path,embeddings_limit)
        classifier = model_dict['classifier']
        max_len = None
    classes = model_dict['classifier_classnames']
    return classifier, classes, pretrained_path, pretrained, embeddings_limit, \
           fasttext_model_path, max_len


def test_if_already_loaded(model_path,classifiers_attributes):
    '''
    Check if the embedding of a classifier is already loaded in a previoys
    classifier in order not to load it again and extract the attributes of the classifier
    :param model_path: the classifier path
    :param classifiers_attributes:a list of dictionaries with keys :
    classifier,classes,pretrained_path,pretrained,embeddings_limit,fasttext_model_path.
     Every dictionary refers to a classifier previously loaded.
    :return: the attributes of the classifier (classifier,classes,
    pretrained_path,pretrained,embeddings_limit,fasttext_model_path)
    '''
    model_dict = pickle.load(open(model_path, 'rb'))
    found = False
    if model_dict['classifier_type'] == 'fasttext':
        for classifier in classifiers_attributes:
            if classifier['fasttext_model_path'] == \
                    model_dict['fasttext_model']:
                fasttext_model_path = model_dict['fasttext_model']
                print("--> Copying the fasttext model from previous loading")
                classifier = classifiers_attributes['classifier']
                embeddings_limit = None
                pretrained = None
                pretrained_path = None
                classes = model_dict['classifier_classnames']
                max_len = None
                found = True
                break
    elif model_dict['embedding_model'] != 'bert':
        for classifier in classifiers_attributes:
            if classifier['embeddings_limit'] == model_dict['embeddings_limit'] \
                    and classifier['pretrained_path'] == \
                    model_dict['embedding_model']:
                fasttext_model_path = None
                pretrained_path = model_dict['embedding_model']
                embeddings_limit = model_dict['embeddings_limit']
                print("--> Copying the text embeddings "
                      "model from previous loading")
                pretrained = classifier['pretrained']
                classifier = model_dict['classifier']
                classes = model_dict['classifier_classnames']
                max_len = None
                found = True
                break
    elif model_dict['embedding_model'] == 'bert':
        for classifier in classifiers_attributes:
            if classifier['pretrained_path'] == 'bert':
                fasttext_model_path = None
                pretrained_path = model_dict['embedding_model']
                embeddings_limit = None
                print("--> Copying the bert text embeddings "
                      "from previous loading")
                pretrained = classifier['pretrained']
                classifier = model_dict['classifier']
                classes = model_dict['classifier_classnames']
                max_len = model_dict['max_len']
                found = True
                break
    if not(found):
        classifier, classes, pretrained_path, pretrained, \
        embeddings_limit, fasttext_model_path, max_len = \
            load_text_classifier_attributes(model_path)
    return  classifier, classes, pretrained_path, pretrained, \
                embeddings_limit, fasttext_model_path, max_len


def load_classifiers(text_models_directory):
    classifiers_attributes = []
    for filename in os.listdir(text_models_directory):
        if filename.endswith(".pt"):
            model_path = os.path.join(text_models_directory, filename)
            dictionary = {}
            dictionary['classifier'], dictionary['classes'], \
            dictionary['pretrained_path'], dictionary['pretrained'], \
            dictionary['embeddings_limit'], dictionary['fasttext_model_path'], dictionary['max_len'] = \
                test_if_already_loaded(model_path, classifiers_attributes)
            classifiers_attributes.append(dictionary)
    return classifiers_attributes


def load_text_dataset(data, hop_samples=None):
    """
    Loads a text dataset
    :param data: csv file with one column transcriptions (text samples)
                   and one column labels
    :param hop_samples: number of samples to hop
    :return: 1. transcriptions: list of text segments
             2. labels: list of labels
             3. classnames: the name of the classes
    """
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
    """Return a mapping from folder to class and
    a mapping from class to folder."""
    folder2idx = {}
    idx2folder = {}
    for idx, folder in enumerate(folders):
        folder2idx[folder] = idx
        idx2folder[idx] = folder
    return folder2idx, idx2folder


def convert_to_fasttext_data(labels, transcriptions):
    """
    Converts data in the correct form to use in fasttext training.
    :param labels: list of string labels written in the form __label__name
    :param transcriptions: transcriptions: list of text segments
    :param filename: file to save the output data
    """
    data = []
    for label, trans in zip(labels, transcriptions):
        trans_pre = text_preprocess(trans)
        data.append("__label__" + label + " " + trans_pre)

    num_of_training_samples = int(0.8 * len(labels))
    df = pd.DataFrame(data[0:num_of_training_samples - 1])
    df.to_csv("train.txt", index=False, sep=' ',
              header=None, quoting=csv.QUOTE_NONE,
              quotechar="", escapechar=" ")
    df = pd.DataFrame(data[num_of_training_samples:-1])
    df.to_csv("test.txt", index=False, sep=' ',
              header=None, quoting=csv.QUOTE_NONE,
              quotechar="", escapechar=" ")


def max_sentence_length(sentences):
    lengths = [len(sentence.split()) for sentence in sentences]
    mu = np.mean(lengths)
    std = np.std(lengths)
    maximum = max(lengths)

    print("----> Documents' lengths: max = {}, mean = {}, std = {}".format(
        maximum, mu, std))

    forced_length = int(mu + std)
   
    return maximum


def seed_torch():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True


def bert_dataframe(sentences, labels):
    d = {'sentence': sentences, 'label': labels}
    print(d)
    df = pd.DataFrame(d)
    return df


def bert_preprocessing(sentences, labels):

    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)

    df = bert_dataframe(sentences, labels)

    max_len = max_sentence_length(sentences)

    return df, le, max_len


def check_balance(labels):
    """
    Checks if a dataset is imbalanced
    :param labels: list of labels
    :return: True if imbalanced, False otherwise
    """
    class_samples = Counter(labels)
    balanced_degrees = 1/len(class_samples)
    imbalanced_degrees = {}
    for label in class_samples:
        imbalanced_degrees[label] = class_samples[label]/len(labels)
    print('--> Dataset samples per class: \n    {}'.format(
        imbalanced_degrees))

    is_imbalanced = False
    for label in imbalanced_degrees:
        if imbalanced_degrees[label] < 0.8 * balanced_degrees:
            is_imbalanced = True
            print("--> The class {} is undersampled.".format(label))

    return is_imbalanced


def split_data(x, y, test_size=0.2, fasttext_data=False, seed=None):
    """
    Split data to train and test in a stratified manner.
    :param x: feature matrix
    :param y: list of labels
    :param test_size: percentage of the test set
    :param fasttext_data: True if the data arre in the fasttext form
    :param seed: seed
    :return: 1. x_train: train feature matrix
             2. x_test: test feature matrix
             3. y_train: train labels
             4. y_test: test labels
    """

    if fasttext_data:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=seed, stratify=y)

        convert_to_fasttext_data(y_train, x_train, 'train.txt')
        convert_to_fasttext_data(y_test, x_test, 'test.txt')
        print("Splitted dataset into train (saved on train.txt) "
              "and test (saved on test.txt) subsets.")
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=seed, stratify=y)
        return x_train, x_test, y_train, y_test


def save_model(model_dict,out_folder, out_model=None, name=None):
    """
    Saves a model dictionary
    :param model_dict: model dictionary
    :param out_model: path to save the model (optional)
    :param name: name of the model (optional)
    :param is_text: True if the model is a text classifier
                    False if it is an audio classifier
    """
    script_dir = os.path.dirname(__file__)

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


def plot_feature_histograms(list_of_feature_mtr, feature_names,
                            class_names, n_columns=5):
    '''
    Plots the histograms of all classes and features for a given
    classification task.
    :param list_of_feature_mtr: list of feature matrices
                                (n_samples x n_features) for each class
    :param feature_names:       list of feature names
    :param class_names:         list of class names, for each feature matr
    '''
    n_features = len(feature_names)
    n_bins = 12
    n_rows = int(n_features / n_columns) + 1
    figs = plotly_sub.make_subplots(rows=n_rows, cols=n_columns,
                                      subplot_titles=feature_names)
    figs['layout'].update(height=(n_rows * 250))
    clr = get_color_combinations(len(class_names))
    for i in range(n_features):
        # for each feature get its bin range (min:(max-min)/n_bins:max)
        f = np.vstack([x[:, i:i + 1] for x in list_of_feature_mtr])
        unique_value = False
        if f.max() == f.min():
            unique_value = True
            value = f.min()
        else:
            bins = np.arange(f.min(), f.max(), (f.max() - f.min()) / n_bins)
        for fi, f in enumerate(list_of_feature_mtr):
            # load the color for the current class (fi)
            mark_prop = dict(color=clr[fi], line=dict(color=clr[fi], width=3))
            # compute the histogram of the current feature (i) and normalize:
            if unique_value == False:
                h, _ = np.histogram(f[:, i], bins=bins)
                h = h.astype(float) / h.sum()
                cbins = (bins[0:-1] + bins[1:]) / 2 
            else:
                h=[1.0]
                cbins = [value]
            scatter_1 = go.Scatter(x=cbins, y=h, name=class_names[fi],
                                   marker=mark_prop, showlegend=(i == 0))
            # (show the legend only on the first line)
            figs.append_trace(scatter_1, int(i/n_columns)+1, i % n_columns+1)
    for i in figs['layout']['annotations']:
        i['font'] = dict(size=10, color='#224488')
    plotly.offline.plot(figs, filename="report.html", auto_open=True)


def get_color_combinations(n_classes):
    clr_map = plt.cm.get_cmap('jet')
    range_cl = range(int(int(255/n_classes)/2), 255, int(255/n_classes))
    clr = []
    for i in range(n_classes):
        clr.append('rgba({},{},{},{})'.format(clr_map(range_cl[i])[0],
                                              clr_map(range_cl[i])[1],
                                              clr_map(range_cl[i])[2],
                                              clr_map(range_cl[i])[3]))
    return clr


def make_group_list(filenames):
    '''
    This function is responsible for creating group id of every sample according to the speaker
    :param filenames: list of samples' names
    :return: list of group ids
    '''
    groups_id = []
    groups_name =[]
    id = 1
    for f in filenames:
        user_name = f.split('/')
        user_name = user_name[-1].split('_')
        user_name = user_name[1]
        found = False
        for count, gr in enumerate(groups_id):
            if user_name == groups_name[count]:
                groups_id.append(gr)
                groups_name.append(user_name)
                found = True
                break
        if found == False:
            groups_id.append(id)
            groups_name.append(user_name)
            id += 1
    return groups_id


def grid_init(clf, clf_name, parameters_dict,
              is_imbalanced, scoring, refit, seed=None, group=False):
    """
    Initializes a grid using:
        1. a pipeline containing:
            - a sampler if the dataset is imbalanced
            - a scaler
            - a variance threshold
            - pca
            - classifier
        2. randomized stratified cross validation using RepeatedStratifiedKFold
        3. dictionary of parameters to be optimized
        4. GridSearchCV
    :param clf: classifier
    :param clf_name: classifier name
    :param parameters_dict: dictionary of parameters to be optimized
    :param is_imbalanced: True if the dataset is imbalanced, False otherwise
    :param seed: seed
    :param filenames: None if no independent split needed, filenames of samples for independent-speaker split
    :return: initialized grid
    """
    if is_imbalanced:
        print('--> The dataset is imbalanced. SMOTETomek will'
              ' be applied to balance the classes')
        sampler = SMOTETomek(random_state=seed)

    scaler = StandardScaler()

    thresholder = VarianceThreshold(threshold=0)

    pca = PCA()
    if is_imbalanced:
        pipe = Pipeline(steps=[
            ('sampling', sampler), ('scaler', scaler),
            ('thresholder', thresholder), ('pca', pca),
            (clf_name, clf)], memory='sklearn_tmp_memory')

    else:
        pipe = Pipeline(steps=[
            ('scaler', scaler), ('thresholder', thresholder),
            ('pca', pca), (clf_name, clf)], memory='sklearn_tmp_memory')

    if group:
        cv = LeaveOneGroupOut()
    else:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)

    grid_clf = GridSearchCV(
        pipe, parameters_dict, cv=cv,
        scoring=scoring, refit=refit, n_jobs=1, verbose=2)

    return grid_clf


def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def f1_micro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="micro")


def _count_score(y_true, y_pred, label1=0, label2=1):
    return sum((y == label1 and pred == label2)
                for y, pred in zip(y_true, y_pred))


def grid_combinations(params_dict):
    num_combinations = 1
    for key in params_dict:
        num_combinations = num_combinations * len(params_dict[key])

    return num_combinations


def custom_auc(ground_truth, predictions):
     # I need only one column of predictions["0" and "1"]. You can get an error here
     # while trying to return both columns at once
     fpr, tpr, _ = roc_curve(ground_truth, predictions, pos_label=1)
     global y_true
     global y_pred
     y_true += list(ground_truth)
     y_pred += list(predictions)
     return auc(fpr, tpr)


def num_group_splits(groups):
    logo = LeaveOneGroupOut()
    return logo.get_n_splits(groups=groups)


def best_clf(grid, labels_set, num_splits, num_models, is_imbalanced, seed=None):

    f1_scores = []

    for model_idx in range(num_models):
        confusion = defaultdict(lambda: defaultdict(int))
        for label1 in labels_set:
            for label2 in labels_set:
                for i in range(num_splits):
                    key = 'split%s_test_count_%s_%s' % (i, label1, label2)
                    val = int(grid.cv_results_[key][model_idx])
                    confusion[label1][label2] += val
                    #calculate the number of all the test samples across the test folds
        cm = {key: dict(value) for key, value in confusion.items()}

        cm = pd.DataFrame.from_dict(cm, orient='index')

        cm = cm.to_numpy()
        f1_0 = get_f1_score(cm, 0)
        f1_1 = get_f1_score(cm, 1)
        mean_f1_from_cm = (f1_0 + f1_1) / 2
        f1_scores.append(mean_f1_from_cm)

    print("\n--> F1 scores from GridSearch:\n {}".format(f1_scores))
    f1_scores = np.array(f1_scores)
    f1_scores = np.nan_to_num(f1_scores)
    best_idx = np.argmax(f1_scores)
    best_score = np.max(f1_scores)
    print("\n--> F1 score of the best classifier (measured on aggregated cm): {}".format(best_score))

    best_params = grid.cv_results_["params"][best_idx]

    print("--> Parameters of best model: {}\n".format(best_params))

    if is_imbalanced:
        sampler = SMOTETomek(random_state=seed)

    scaler = StandardScaler()

    thresholder = VarianceThreshold(threshold=0)

    pca = PCA(n_components=best_params["pca__n_components"])
    clf =  svm.SVC(
        kernel="rbf", gamma=best_params["SVM_RBF__gamma"],
        C=best_params["SVM_RBF__C"], probability=True,
        class_weight='balanced')
    if is_imbalanced:
        pipe = Pipeline(steps=[
            ('sampling', sampler), ('scaler', scaler),
            ('thresholder', thresholder), ('pca', pca),
            ("SVM", clf)], memory='sklearn_tmp_memory')

    else:
        pipe = Pipeline(steps=[
            ('scaler', scaler), ('thresholder', thresholder),
            ('pca', pca), ("SVM", clf)], memory='sklearn_tmp_memory')

    return pipe


def print_grid_results(grid, metric, labels_set, num_splits):

    clf_params = grid.best_params_
    clf_score = grid.best_score_
    clf_stdev = grid.cv_results_['std_test_{}'.format(metric)][grid.best_index_]
    print("--> Parameters of best model: {}".format(clf_params))
    print("--> Best validation score:      {:0.5f} (+/-{:0.5f})".format(
        clf_score, clf_stdev))

    best_index = grid.best_index_
    confusion = defaultdict(lambda: defaultdict(int))
    for label1 in labels_set:
        for label2 in labels_set:
            for i in range(num_splits):
                key = 'split%s_test_count_%s_%s' % (i, label1, label2)
                val = int(grid.cv_results_[key][best_index])
                confusion[label1][label2] += val
                #calculate the number of all the test samples across the test folds
    confusion = {key: dict(value) for key, value in confusion.items()}

    confusion = pd.DataFrame.from_dict(confusion,orient='index')

    print("--> Confusion matrix of the best classifier (sum across all tests):")
    print(confusion)

    auc = 0
    for i in range(num_splits):
        key1 = 'split%s_test_auc_1' % (i)
        #summarize all the roc values of all test sets
        auc += grid.cv_results_[key1][best_index]
    #take mean auc value over all the test sets
    auc = auc/num_splits
    print("\nMEAN AUC FOR CLASS 1 ACROSS ALL TESTS OF ALL GRIDSEARCHES: {}".format(auc))
    # make graphics of confusion matrix, class-wise performance measures and roc curves for every positive class

    return auc, confusion


def print_cross_val_results(scores, metric, labels_set, num_splits):

    confusion = defaultdict(lambda: defaultdict(int))
    num_test_samples = [0] * num_splits
    for label1 in labels_set:
        for label2 in labels_set:
            array_of_counts = scores['test_count_%s_%s' % (label1, label2)]
            for i in range(num_splits):
                list_of_counts = list(array_of_counts)
                val = int(list_of_counts[i])
                confusion[label1][label2] += val
                # calculate the number of all the test samples across the test folds
                num_test_samples[i] += val
    confusion = {key: dict(value) for key, value in confusion.items()}

    confusion = pd.DataFrame.from_dict(confusion, orient='index')

    list_of_f1 = list(scores['test_{}'.format(metric)])
    weighted_f1 = [f*num for f,num in zip(list_of_f1,num_test_samples)]
    mean_f1 = sum(weighted_f1)/sum(num_test_samples)
    list_of_aucs = list(scores['test_auc_1'])
    # summarize all the roc values of all test sets
    auc = sum(list_of_aucs)
    # take mean auc value over all the test sets
    auc = auc / num_splits
    return auc, confusion


def make_graphics(cm, mean_f1):

    global y_true
    global y_pred
    auc = roc_auc_score(y_true, y_pred)
    print("\n--> AUC COMPUTED FROM CONCATENATED POSTERIORS: {}".format(auc))
    titles = ["Confusion matrix, Mean F1 (macro): {:.1f}%".format(100 * mean_f1),
              "Class-wise Performance measures",
              "Pre vs Rec for positive",
              "ROC for positive, AUC: {}".format(auc)]
    rec_c, pre_c, f1_c = aT.compute_class_rec_pre_f1(cm)
    figs = plotly.subplots.make_subplots(rows=2, cols=2,
                                         subplot_titles=titles)

    heatmap = go.Heatmap(z=np.flip(cm, axis=0), x=["negative","positive"],
                         y=list(reversed(["negative","positive"])),
                         colorscale=[[0, '#4422ff'], [1, '#ff4422']],
                         name="confusion matrix", showscale=False)
    mark_prop1 = dict(color='rgba(80, 220, 150, 0.5)',
                      line=dict(color='rgba(80, 220, 150, 1)', width=2))
    mark_prop2 = dict(color='rgba(80, 150, 220, 0.5)',
                      line=dict(color='rgba(80, 150, 220, 1)', width=2))
    mark_prop3 = dict(color='rgba(250, 150, 150, 0.5)',
                      line=dict(color='rgba(250, 150, 150, 1)', width=3))
    b1 = go.Bar(x=["negative","positive"], y=rec_c, name="Recall", marker=mark_prop1)
    b2 = go.Bar(x=["negative","positive"], y=pre_c, name="Precision", marker=mark_prop2)
    b3 = go.Bar(x=["negative","positive"], y=f1_c, name="F1", marker=mark_prop3)

    figs.append_trace(heatmap, 1, 1);
    figs.append_trace(b1, 1, 2)
    figs.append_trace(b2, 1, 2);
    figs.append_trace(b3, 1, 2)


    pre, rec, thr_prre = precision_recall_curve(y_true,y_pred)
    fpr, tpr, thr_roc = roc_curve(y_true, y_pred)
    figs.append_trace(go.Scatter(x=thr_prre, y=pre, name="Precision",
                                     marker=mark_prop1), 2, 1)
    figs.append_trace(go.Scatter(x=thr_prre, y=rec, name="Recall",
                                 marker=mark_prop2), 2, 1)
    figs.append_trace(go.Scatter(x=fpr, y=tpr, showlegend=False), 2, 2)
    figs.update_xaxes(title_text="threshold", row=2, col=1)
    figs.update_xaxes(title_text="false positive rate", row=2, col=2)
    figs.update_yaxes(title_text="true positive rate", row=2, col=2)

    plotly.offline.plot(figs, filename="figs.html", auto_open=True)


def get_f1_score(confusion_matrix, i):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for j in range(len(confusion_matrix)):
        if (i == j):
            TP += confusion_matrix[i, j]
            tmp = np.delete(confusion_matrix, i, 0)
            tmp = np.delete(tmp, j, 1)

            TN += np.sum(tmp)
        else:
            if (confusion_matrix[i, j] != 0):

                FN += confusion_matrix[i, j]
            if (confusion_matrix[j, i] != 0):

                FP += confusion_matrix[j, i]

    recall = TP / (FN + TP)
    precision = TP / (TP + FP)
    f1_score = 2 * 1/(1/recall + 1/precision)

    return f1_score


def train_basic_segment_classifier(feature_matrix, labels,
                                   is_imbalanced, config, seed=None):
    """
    Trains basic (i.e. svm or xgboost) classifier pipeline
    :param feature_matrix: feature matrix
    :param labels: list of labels
    :param is_imbalanced: True if the dataset is imbalanced, False otherwise
    :param config: configuration file
    :param seed: seed
    :return: the trained pipeline
    """

    n_components = [0.98, 0.99, 'mle', None]

    if config['metric'] == 'f1_macro':
        metric = make_scorer(f1_macro)
    elif config['metric'] == 'f1_micro':
        metric = make_scorer(f1_micro)
    elif config['metric'] == 'accuracy':
        metric = make_scorer(accuracy_score)
    else:
        print('Only supported evaluation metrics are: f1_macro, f1_micro, accuracy')
        return -1

    labels_set = sorted(set(labels))
    scorer = {}
    for label1 in labels_set:
        for label2 in labels_set:
            count_score = make_scorer(_count_score, label1=label1,
                                      label2=label2)
            scorer['count_%s_%s' % (label1, label2)] = count_score

    scorer[config['metric']] = metric
    auc = make_scorer(custom_auc, greater_is_better=True, needs_proba=True)
    scorer['auc_1'] = auc
    if config['svm']:
        clf = svm.SVC(kernel=config['svm_parameters']['kernel'], probability=True,
                      class_weight='balanced')
        svm_parameters = {'gamma': ['auto', 'scale'],
                          'C': [0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0]}

        parameters_dict = dict(pca__n_components=n_components,
                               SVM__gamma=svm_parameters['gamma'],
                               SVM__C=svm_parameters['C'])

        grid_clf = grid_init(clf, "SVM", parameters_dict,
                             is_imbalanced, scoring=scorer,
                             refit=config['metric'], seed=seed)
        print("--> Training SVM classifier using GridSearchCV")

    elif config['xgboost']:
        xgb = XGBClassifier(n_estimators=100)
        parameters_dict = dict(pca__n_components=n_components)
        grid_clf = grid_init(xgb, "XGBOOST", parameters_dict,
                             is_imbalanced, scoring=scorer,
                             refit=config['metric'], seed=seed)
        print("--> Training XGBOOST classifier using GridSearchCV")

    else:
        print("The only supported basic classifiers are SVM and XGBOOST")
        return -1

    grid_clf.fit(feature_matrix, labels)

    clf_svc = grid_clf.best_estimator_

    num_splits = 5 * 3
    _, _ = print_grid_results(grid_clf, config['metric'], labels_set,num_splits)

    return clf_svc


def train_recording_level_classifier(feature_matrix, labels,
                                     is_imbalanced, config, filenames, seed=None):
    """
        Trains basic (i.e. svm,svm rbf,gradientboosting,knn,
        randomforest,extratrees) classifier pipeline
        :param feature_matrix: feature matrix
        :param labels: list of labels
        :param is_imbalanced: True if the dataset is imbalanced, False otherwise
        :param config: configuration file
        :param seed: seed
        :return: the trained pipeline
        """

    if config['metric'] == 'f1_macro':
        metric = make_scorer(f1_macro)
    elif config['metric'] == 'f1_micro':
        metric = make_scorer(f1_micro)
    elif config['metric'] == 'accuracy':
        metric = make_scorer(accuracy_score)
    else:
        print('Only supported evaluation metrics are: f1_macro, f1_micro, accuracy')
        return -1

    labels_set = sorted(set(labels))
    scorer = {}
    for label1 in labels_set:
        for label2 in labels_set:
            count_score = make_scorer(_count_score, label1=label1,
                                      label2=label2)
            scorer['count_%s_%s' % (label1, label2)] = count_score

    scorer[config['metric']] = metric

    n_components = [0.98, 0.99, None]


    if config['classifier_type'] == 'svm_rbf':
        clf = svm.SVC(kernel='rbf', probability=True,
                      class_weight='balanced')
        svm_parameters = {'gamma': ['auto', 'scale'],
                          'C': [0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0]}

        parameters_dict = dict(pca__n_components=n_components,
                               SVM_RBF__gamma=svm_parameters['gamma'],
                               SVM_RBF__C=svm_parameters['C'])

        grid_clf = grid_init(clf, "SVM_RBF", parameters_dict,
                             is_imbalanced, scoring=scorer,
                             refit=config['metric'], seed=seed, group=True)
        print("--> Training SVM rbf classifier using GridSearchCV")
    elif config['classifier_type'] == 'svm':
        clf = svm.SVC(class_weight='balanced')
        svm_parameters = {'gamma': ['auto', 'scale'],
                          'C': [1e-1, 1, 5, 1e1]}

        parameters_dict = dict(pca__n_components=n_components,
                               SVM__gamma=svm_parameters['gamma'],
                               SVM__C=svm_parameters['C'])

        grid_clf = grid_init(clf, "SVM", parameters_dict,
                             is_imbalanced, scoring=scorer,
                             refit=config['metric'], seed=seed, group=True)
        print("--> Training SVM classifier using GridSearchCV")
    elif config['classifier_type'] == 'randomforest':
        clf = RandomForestClassifier()
        classifier_parameters = {'n_estimators' :[10, 25, 50, 100, 200, 500]}

        parameters_dict = dict(pca__n_components=n_components,
                               RandomForest__n_estimators=
                               classifier_parameters['n_estimators'])

        grid_clf = grid_init(clf, "RandomForest", parameters_dict,
                             is_imbalanced, scoring=scorer,
                             refit=config['metric'], seed=seed, group=True)
        print("--> Training Random Forest classifier using GridSearchCV")
    elif config['classifier_type'] == 'knn':
        clf = KNeighborsClassifier()
        classifier_parameters = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]}
        parameters_dict = dict(pca__n_components=n_components,
                               Knn__n_neighbors=classifier_parameters['n_neighbors'])
        grid_clf = grid_init(clf, "Knn", parameters_dict,
                             is_imbalanced, scoring=scorer,
                             refit=config['metric'], seed=seed, group=True)
        print("--> Training Knn classifier using GridSearchCV")
    elif config['classifier_type'] == 'gradientboosting':
        clf = GradientBoostingClassifier()
        classifier_parameters = {'n_estimators': [10, 25, 50, 100, 200, 500]}

        parameters_dict = dict(pca__n_components=n_components,
                               GradientBoosting__n_estimators=classifier_parameters['n_estimators'])

        grid_clf = grid_init(clf, "GradientBoosting", parameters_dict,
                             is_imbalanced, scoring=scorer,
                             refit=config['metric'], seed=seed, group=True)
        print("--> Training Gradient Boosting classifier using GridSearchCV")
    elif config['classifier_type'] == 'extratrees':
        clf = ExtraTreesClassifier()
        classifier_parameters = {'n_estimators': [10, 25, 50, 100, 200, 500]}

        parameters_dict = dict(pca__n_components=n_components,
                               Extratrees__n_estimators=classifier_parameters['n_estimators'])

        grid_clf = grid_init(clf, "Extratrees", parameters_dict,
                             is_imbalanced, scoring=scorer,
                             refit=config['metric'], seed=seed, group=True)
        print("--> Training Extra Trees classifier using GridSearchCV")

    labels_set = sorted(set(labels))

    groups = make_group_list(filenames)

    #use gridsearch to find best parameters
    grid_clf.fit(feature_matrix, labels, groups=groups)
    num_combos = grid_combinations(parameters_dict)
    num_splits = num_group_splits(groups)
    clf_svc = best_clf(
        grid_clf, labels_set, num_splits,
        num_combos, is_imbalanced, seed)

    #use best parameters to cross validate again and make figures/compute scores

    scorer = {}
    scorer[config['metric']] = metric
    for label1 in labels_set:
        for label2 in labels_set:
            count_score = make_scorer(_count_score, label1=label1,
                                      label2=label2)
            scorer['count_%s_%s' % (label1, label2)] = count_score

    auc = make_scorer(custom_auc, greater_is_better=True, needs_proba=True)
    scorer['auc_1'] = auc

    cv = LeaveOneGroupOut()
    scores = cross_validate(
        clf_svc, feature_matrix, y=labels,
        groups=groups, scoring=scorer, cv=cv)

    auc_total, cm = print_cross_val_results(scores, config['metric'], labels_set, num_splits)
    print("\n--> AGGREGATED CONFUSION MATRIX:")
    print(cm)
    cm = cm.to_numpy()
    f1_0 = get_f1_score(cm, 0)
    f1_1 = get_f1_score(cm, 1)
    mean_f1_from_cm = (f1_0 + f1_1) / 2
    
    # print the mean auc value for positive class
    print("\n--> MEAN AUC FOR CLASS 1 ACROSS ALL TESTS OF ALL GRIDSEARCHES: {}".format(auc_total))
    # make graphics of confusion matrix, class-wise performance measures and roc curves for every positive class

    make_graphics(cm, mean_f1_from_cm)
    clf_svc.fit(feature_matrix, y=labels)
    return clf_svc
