import os
import numpy as np
import time
import argparse
import yaml
import fasttext
from gensim.models import KeyedVectors
from feature_extraction import TextFeatureExtraction
from utils import load_text_dataset, check_balance,\
    convert_to_fasttext_data, save_model,\
    train_basic_segment_classifier

script_dir = os.path.dirname(__file__)
eps = np.finfo(float).eps
seed = 500
if not script_dir:
    with open(r'./config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
else:
    with open(script_dir + '/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

config = config['text_classifier']


def basic_segment_classifier(data, feature_extractor, out_model):
    """
    Reads the config file and trains a classifier. It stores a model_dict
    containing the following information:
        - classifier: the output classifier
        - classifier_type: the type of the output classifier (i.e basic)
        - classifier_classnames: the name of the classes
        - embedding_model: path to the used embeddings model
        - embeddings_limit: possible limit on embeddings vectors

    :param data: csv file with one column transcriptions (text samples)
                   and one column labels
    :param feature_extractor: an initialized TextFeatureExtraction object
    :param out_model: name of the output model
    :return None
    """

    np.random.seed(seed)

    class_file_name = out_model + "_classenames.csv"

    print('--> Loading Dataset...')
    transcriptions, labels, classnames = load_text_dataset(
        data, config['hop_samples'])

    is_imbalanced = check_balance(labels)

    # extract features based on pretrained fasttext model

    total_features = feature_extractor.transform(transcriptions)

    clf = train_basic_segment_classifier(
        total_features, labels, is_imbalanced, config, seed)

    model_dict = {}
    model_dict['classifier_type'] = 'basic'
    model_dict['classifier'] = clf
    model_dict['classifier_classnames'] = classnames
    model_dict['embedding_model'] = feature_extractor.embedding_model
    model_dict['embeddings_limit'] = feature_extractor.embeddings_limit

    if out_model is None:
        save_model(model_dict, name="basic_classifier")
    else:
        save_model(model_dict, out_model=out_model)

    return None


def train_fasttext_segment_classifier(data, embeddings_limit, out_model):
    """
        Reads the config file and uses the pretrained_embedding_vectors
        variable to train a classifier. It stores the
        generated classifier with name fasttext_classifier_{time}.ftz
        and a model_dict containing the following information:
            - fasttext_model: path to the output classifier
            - classifier_type: the type of the output classifier (i.e fasttext)
            - classifier_classnames: the name of the classes

        :param data: csv file with one column transcriptions (text samples)
                    and one column labels with labels written as __label__name
                    (eg. __label__act_low)
        :param embeddings_limit: a possible limit to the embeddings vectors
        :param out_model: name of the output model
        :return None
        """

    np.random.seed(seed)

    class_file_name = out_model + "_classenames.csv"

    pretrained_embedding_vectors = config['pretrained_embedding_vectors']
    print("--> Loading the text embeddings model")
    word_model = KeyedVectors.load_word2vec_format(
        pretrained_embedding_vectors, limit=embeddings_limit)
    word_model.save_word2vec_format('tmp.vec')
    print('--> Loading Dataset...')
    transcriptions, labels, classnames = load_text_dataset(
        data, config['hop_samples'])

    convert_to_fasttext_data(labels, transcriptions, 'train.txt')

    print("--> Training classifier using fasttext")
    model = fasttext.train_supervised(
        input='train.txt', epoch=25, lr=1.0,
        wordNgrams=2, verbose=2, minCount=1,
        loss="hs", dim=300, pretrainedVectors='tmp.vec',
        seed=seed)

    timestamp = time.ctime()
    name = "fasttext_classifier_{}.ftz".format(timestamp)
    out_folder = config["out_folder"]
    if not script_dir:
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        out_path = os.path.join(out_folder, name)
    else:
        out_folder = os.path.join(script_dir, out_folder)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        out_path = os.path.join(out_folder, name)

    model.save_model(out_path)
    model_dict = {}
    model_dict['classifier_type'] = 'fasttext'
    model_dict['fasttext_model'] = out_path
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

    if config['fasttext']:
        train_fasttext_segment_classifier(
            args.annotation, args.embeddings_limit, args.outputmodelpath)

    elif config['svm'] or config['xgboost']:
        feature_extractor = TextFeatureExtraction(args.pretrained,
                                                  args.embeddings_limit)
        basic_segment_classifier(args.annotation, feature_extractor,
                          args.outputmodelpath)
    else:
        print('SVM and fasttext are the only supported classifiers.')
