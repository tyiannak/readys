import os
import numpy as np
import time
import argparse
import yaml
import fasttext
from gensim.models import KeyedVectors
from feature_extraction import TextFeatureExtraction
from utils import load_dataset, check_balance, convert_to_fasttext_data, save_model, train_basic_segment_classifier

script_dir = os.path.dirname(__file__)
eps = np.finfo(float).eps
seed = 500
if not script_dir:
    with open(r'./config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
else:
    with open(script_dir + '/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)


def basic_segment_classifier(data, feature_extractor, out_model):
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

    clf = train_basic_segment_classifier(total_features, labels, is_imbalanced, config, seed)

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

    timestamp = time.ctime()
    name = "fasttext_classifier_{}.ftz".format(timestamp)
    out_folder = config['text_classifier']["out_folder"]
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

    if config['text_classifier']['fasttext']:
        train_fasttext_segment_classifier(args.annotation, args.embeddings_limit, args.outputmodelpath)

    elif config['text_classifier']['svm'] or config['text_classifier']['xgboost']:
        feature_extractor = TextFeatureExtraction(args.pretrained,
                                                  args.embeddings_limit)
        basic_segment_classifier(args.annotation, feature_extractor,
                          args.outputmodelpath)
    else:
        print('SVM and fasttext are the only supported classifiers.')
