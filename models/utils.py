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
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit, GroupShuffleSplit
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

fpr0_total = []
tpr0_total = []
fpr1_total = []
tpr1_total = []

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
        bins = np.arange(f.min(), f.max(), (f.max() - f.min()) / n_bins)
        for fi, f in enumerate(list_of_feature_mtr):
            # load the color for the current class (fi)
            mark_prop = dict(color=clr[fi], line=dict(color=clr[fi], width=3))
            # compute the histogram of the current feature (i) and normalize:
            h, _ = np.histogram(f[:, i], bins=bins)
            h = h.astype(float) / h.sum()
            cbins = (bins[0:-1] + bins[1:]) / 2
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
        cv = GroupShuffleSplit(n_splits=5)
    else:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)

    grid_clf = GridSearchCV(
        pipe, parameters_dict, cv=cv,
        scoring=scoring, refit=refit, n_jobs=1)

    return grid_clf


def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def _count_score(y_true, y_pred, label1=0, label2=1):
    return sum((y == label1 and pred == label2)
                for y, pred in zip(y_true, y_pred))

# define scoring function

def custom_auc(ground_truth, predictions,pos_label=1):
     # I need only one column of predictions["0" and "1"]. You can get an error here
     # while trying to return both columns at once
     fpr, tpr, _ = roc_curve(ground_truth, predictions[:, pos_label], pos_label=pos_label)
     print(predictions)
     '''
     if pos_label==0:
        global fpr0_total
        global tpr0_total
        fpr0_total.append(fpr)
        tpr0_total.append(tpr)
     else:
         global fpr1_total
         global tpr1_total
         fpr1_total.append(fpr)
         tpr1_total.append(tpr)
    '''
     return auc(fpr, tpr)

def print_grid_results(grid, metric, labels_set, num_splits):

    clf_params = grid.best_params_
    clf_score = grid.best_score_
    clf_stdev = grid.cv_results_['std_test_{}'.format(metric)][grid.best_index_]
    print("--> Parameters of best svm model: {}".format(clf_params))
    print("--> Best validation score:      {:0.5f} (+/-{:0.5f})".format(
        clf_score, clf_stdev))

    best_index = grid.best_index_
    confusion = defaultdict(lambda: defaultdict(int))
    num_test_samples = 0
    for label1 in labels_set:
        for label2 in labels_set:
            for i in range(num_splits):
                key = 'split%s_test_count_%s_%s' % (i, label1, label2)
                print(grid.cv_results_[key][best_index])
                val = int(grid.cv_results_[key][best_index])
                confusion[label1][label2] += val
                #calculate the number of all the test samples across the test folds
                num_test_samples += val
    confusion = {key: dict(value) for key, value in confusion.items()}

    confusion = pd.DataFrame.from_dict(confusion,orient='index')

    print("--> Confusion matrix of the best classifier (sum across all tests):")
    print(confusion)
    auc_total = []
    for label in labels_set:
        auc = 0
        for i in range(num_splits):
            key1 = 'split%s_test_auc_%s' % (i,label)
            #summarize all the roc values of all test sets
            auc += grid.cv_results_[key1][best_index]
        #take mean auc value over all the test sets
        auc = auc/num_splits
        #save auc mean value computed on positive class = current label
        auc_total.append(auc)
    return num_test_samples, auc_total, confusion


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
    elif config['metric'] == 'accuracy':
        metric = make_scorer(accuracy_score)
    else:
        print('Only supported evaluation metrics are: f1_macro, accuracy')
        return -1

    labels_set = sorted(set(labels))
    scorer = {}
    for label1 in labels_set:
        for label2 in labels_set:
            count_score = make_scorer(_count_score, label1=label1,
                                      label2=label2)
            scorer['count_%s_%s' % (label1, label2)] = count_score

    scorer[config['metric']] = metric
    for label in labels_set:
        fpr, tpr = make_scorer(custom_auc,pos_label=label, needs_proba=True)
        scorer['roc_fpr_%s' % label] = fpr
        scorer['roc_tpr_%s' % label] = tpr
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
    _,_,_ = print_grid_results(grid_clf, config['metric'], labels_set,num_splits)
    clf_svc.fit(feature_matrix, labels)
    return clf_svc

def make_graphics(cm,mean_f1):

    titles = ["Confusion matrix, Mean F1 (macro): {:.1f}%".format(100 * mean_f1),
              "Class-wise Performance measures",
              "ROC for class 0",
              "ROC for class 1"]
    rec_c, pre_c, f1_c = aT.compute_class_rec_pre_f1(cm)
    figs = plotly.subplots.make_subplots(rows=2, cols=2,
                                         subplot_titles=titles)

    heatmap = go.Heatmap(z=np.flip(cm, axis=0), x=cm,
                         y=[0,1],
                         colorscale=[[0, '#4422ff'], [1, '#ff4422']],
                         name="confusion matrix", showscale=False)
    mark_prop1 = dict(color='rgba(80, 220, 150, 0.5)',
                      line=dict(color='rgba(80, 220, 150, 1)', width=2))
    mark_prop2 = dict(color='rgba(80, 150, 220, 0.5)',
                      line=dict(color='rgba(80, 150, 220, 1)', width=2))
    mark_prop3 = dict(color='rgba(250, 150, 150, 0.5)',
                      line=dict(color='rgba(250, 150, 150, 1)', width=3))
    b1 = go.Bar(x=[0,1], y=rec_c, name="Recall", marker=mark_prop1)
    b2 = go.Bar(x=[0,1], y=pre_c, name="Precision", marker=mark_prop2)
    b3 = go.Bar(x=[0,1], y=f1_c, name="F1", marker=mark_prop3)

    figs.append_trace(heatmap, 1, 1);
    figs.append_trace(b1, 1, 2)
    figs.append_trace(b2, 1, 2);
    figs.append_trace(b3, 1, 2)

    #compute the mean false positive rates and true positive rates over all of 15 test sets (3*5)
    #for CLASS 0
    global fpr1_total
    global tpr1_total
    final1_fpr = [0] * fpr1_total[0].size()
    final1_tpr = [0] * tpr1_total[0].size()
    for (array1,array2) in zip(fpr1_total,tpr1_total):
        final1_fpr += array1
        final1_tpr += array2
    final1_fpr = final1_fpr/15
    final1_tpr = final1_tpr/15

    # compute the mean false positive rates and true positive rates over all of 15 test sets (3*5)
    # for CLASS 1
    global fpr2_total
    global tpr2_total
    final2_fpr = [0] * len(fpr2_total[0])
    final2_tpr = [0] * len(tpr2_total[0])
    for (array1, array2) in zip(fpr2_total, tpr2_total):
        final2_fpr += array1
        final2_tpr += array2
    final2_fpr = final2_fpr / 15
    final2_tpr = final2_tpr / 15
    figs.append_trace(go.Scatter(x=final1_fpr, y=final1_tpr, showlegend=False), 2, 1)
    figs.append_trace(go.Scatter(x=final2_fpr, y=final2_tpr, showlegend=False), 2, 2)
    figs.update_xaxes(title_text="false positive rate", row=2, col=1)
    figs.update_yaxes(title_text="true positive rate", row=2, col=1)
    figs.update_xaxes(title_text="false positive rate", row=2, col=2)
    figs.update_yaxes(title_text="true positive rate", row=2, col=2)
    plotly.offline.plot(figs, filename="figs.html", auto_open=True)

def repeated_grouped_KFold(feature_matrix, labels, grid_clf, config, groups):

    labels_set = sorted(set(labels))
    print("---> Group indices: \n{}".format(groups))

    # run 3 times 5-Fold cross-val
    clf_scores = []

    num_test_samples = []

    for idx in range(3):
        print("--> {} of {} 5-Fold Cross Val:".format(idx + 1, 3))
        grid_clf.fit(feature_matrix, labels, groups=groups)
        clf_score = grid_clf.best_score_
        clf_scores.append(clf_score)

        # print the group ids of the samples for every train and test split, in order to
        # check if test set incorporates ids that do not appear in train set
        for train, test in grid_clf.cv.split(feature_matrix, labels, groups=groups):
            print('TRAIN: ', train, ' TEST: ', test)
            print([groups[t] for t in train])
            print([groups[t] for t in test])

        num_splits = 5
        num, auc_total, confusion =  print_grid_results(grid_clf, config['metric'], labels_set, num_splits)
        if idx == 0:
            cm_total = confusion.to_numpy()
            auc = auc_total
        else:
            #summarize all cm over the three gridsearches
            cm_total += confusion.to_numpy()
            #summarize all the auc values(for both positive class cases) over the three gridsearches
            auc = [x + y for x, y in zip(auc, auc_total)]
        num_test_samples.append(num)
    #take mean auc across three gridsearches
    auc = [item/3 for item in auc]
    #take the mean cm of all of the gridsearches
    cm_total = cm_total/cm_total.sum()

    #calculate the mean f1 score across all tests of all gridsearches
    test_samples = 0
    test_true = 0
    for (score, length) in zip(clf_scores, num_test_samples):
        test_samples += length
        test_true += score * length

    test_score = test_true / test_samples
    print("\nMEAN F1 MACRO ACROSS ALL TESTS OF ALL GRIDSEARCHES: {}".format(test_score))

    #print the mean auc value for every positive class
    for i in len(auc):
        print("\nMEAN AUC FOR CLASS {} ACROSS ALL TESTS OF ALL GRIDSEARCHES: {}".format(i,auc[i]))
    #make graphics of confusion matrix, class-wise performance measures and roc curves for every positive class
    #make_graphics(cm_total,test_score)
    return grid_clf


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
    elif config['metric'] == 'accuracy':
        metric = make_scorer(accuracy_score)
    else:
        print('Only supported evaluation metrics are: f1_macro, accuracy')
        return -1

    labels_set = sorted(set(labels))
    scorer = {}
    for label1 in labels_set:
        for label2 in labels_set:
            count_score = make_scorer(_count_score, label1=label1,
                                      label2=label2)
            scorer['count_%s_%s' % (label1, label2)] = count_score

    for label in labels_set:
        auc = make_scorer(custom_auc,pos_label=label,greater_is_better=True, needs_proba=True)
        scorer['auc_%s' % label] = auc

    scorer[config['metric']] = metric

    n_components = [0.99]

    if config['classifier_type'] == 'svm_rbf':
        clf = svm.SVC(kernel='rbf', probability=True,
                      class_weight='balanced')
        svm_parameters = {'gamma': ['auto'],
                          'C': [1]}

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


    groups = make_group_list(filenames)
    grid_clf = repeated_grouped_KFold(feature_matrix, labels, grid_clf, config, groups)

    clf_svc = grid_clf.best_estimator_
    clf_svc.fit(feature_matrix, labels)

    return clf_svc
