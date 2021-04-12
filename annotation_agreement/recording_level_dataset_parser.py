import argparse
import pandas as pd
import os
from shutil import copy

def dataset_parser(name,aggregate_annotations,input_data):
    '''
    This function take as input the aggregated annotations and creates a folder-organised dataset
    (each sample located in a subfolder in accordance with the class belongs to)
    :param aggregate_annotations: csv file of winner annotations per sample
    :param input_data: directory which contains all audio files collected
    '''
    data = pd.read_csv(aggregate_annotations)
    if not(os.path.exists('datasets')):
        os.mkdir('datasets')
    full_name = 'datasets/' + name
    if not(os.path.exists(full_name)):
        os.mkdir(full_name)
    for i,k in enumerate(data['Winner_annotation']):
        subfolder_name = full_name + '/' + str(k)
        if not (os.path.exists(subfolder_name)):
            os.mkdir(subfolder_name)
        sample_name = data['Sample_Name'][i]
        src = input_data + '/' + sample_name
        dst = subfolder_name + '/'+ sample_name
        copy(src,dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name_of_classification_task",required=True,
                        help="the name of the classification task e.g. expressive")
    parser.add_argument("-aa", "--aggregate_annotations",required=True,
                        help="the file which contains the winner annotations per sample")
    parser.add_argument("-i", "--input_data", required=True,
                        help="the directory with all audio files collected")
    args = parser.parse_args()
    dataset_parser(args.name_of_classification_task,args.aggregate_annotations,args.input_data)