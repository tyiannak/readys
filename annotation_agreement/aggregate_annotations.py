"""
This script is used to generate the aggregated annotations.
Usage:
1) download annotation csv (aggregate_annotations.csv)
   and list of files (videofiles.txt)
2) run ./data_prep_linux (or mac depending on your OS)
3) run python3 aggregate_annotations.py
4) aggregated.csv is the final annotation file containing aggregated annotations
   and respective confidences. plots folder contains respective plots
"""

import shutil
import os
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def aggregate_annotations(file,class_number):

    data = pd.read_csv(file)

    # Create Dataframe
    aggregation = {'Sample_Name': [],
                   'Winner_annotation': [],
                   'Confidence': [],
                   'Number_annotations': []}
    
    df = pd.DataFrame(aggregation, columns=list(aggregation.keys()))


    if class_number==1:
        num_anot = (pd.crosstab(data.Sample_name, data.Class1))
        conf = (pd.crosstab(data.Sample_name, data.Class1))
    elif class_number==2:
        num_anot = (pd.crosstab(data.Sample_name, data.Class2))
        conf = (pd.crosstab(data.Sample_name, data.Class2))
    else:
        num_anot = (pd.crosstab(data.Sample_name, data.Class3))
        conf = (pd.crosstab(data.Sample_name, data.Class3))

    
    # Number_annotations
    num_anot['sum'] = num_anot.sum(axis=1)
    num_anot = num_anot.reset_index()

    df['Number_annotations'] = num_anot['sum']
       
    # Confidence
    res = conf.div(conf.sum(axis=1), axis=0)*100
    res = res.reset_index()

    # Values to Dataframe
    df['Sample_Name'] = res['Sample_name']
    sav=res['Sample_name']
    res=res.drop(['Sample_name'], axis=1)
    res['Max'] = res.idxmax(axis=1)
    res['max_value'] = res.max(axis=1)

    df['Winner_annotation'] = res["Max"]
    df['Confidence'] = res["max_value"]

    return df


def save_to_csv(df,name):   

    df.to_csv(name, index=False)


def report_annotations(file, class_number,annotators):

    data = pd.read_csv(file)
    df = aggregate_annotations(file,class_number)
    csv_name = 'aggregate_annotations_of_class' + str(class_number) + '.csv'
    save_to_csv(df, csv_name)

    class_column = 'Class' + str(class_number)

    with open('audiofiles.txt') as f:
        vidfiles = f.read().splitlines()

    print("\nTotal files available: ", len(vidfiles))

    sample_num = set(data['Sample_name'])

    # Total files annotated
    print("\nNum of files annotated: ", len(list(set(vidfiles) & sample_num)))

    # Num of files NOT annotated
    print("\nNum of files not annotated: ",
          len(list(set(vidfiles) - sample_num)))

    # Total annotations
    print("\nTotal annotations:", df['Number_annotations'].sum())

    # Create directory for plots, if dir exists delete it
    if not(os.path.exists('plots')):
        os.mkdir('plots')

    # Number of annotation that every user did + plot
    print("\nAnnotations per user:\n", data['Username'].value_counts())
    user = data['Username'].value_counts()
    user.plot(kind='pie', subplots=True, shadow=True,
              startangle=90, figsize=(15, 10), autopct='%1.1f%%')
    pie_name = 'plots/pie' + class_column + '.png'
    plt.savefig(pie_name)
    plt.close()

    # Class distribution (before majority) + plot

    count = data[class_column].value_counts()
    count = count.to_frame()
    per = count.div(count.sum(axis=0))*100

    count['Percentage'] = per[class_column]
    count['Percentage'] = pd.Series([round(val, 2)
                                     for val in count['Percentage']],
                                    index=count.index)
    count['Percentage'] = pd.Series(["{0:.2f}%".format(val)
                                     for val in count['Percentage']],
                                    index=count.index)


    print("\nInitial Class Distribution (before majority): "
            "\nTotal: %s \n%s" % (data[class_column].shape[0], count))



    conf = data[class_column].value_counts()
    conf.plot.bar()
    plt.xlabel('Classes')
    plt.ylabel('Number')
    plt.title('Class distribution')
    plt.tight_layout()
    before_name = 'plots/class_distr_before' + class_column + '.png'
    plt.savefig(before_name)

    # Class distribution (after majority) + plot
    count = df['Winner_annotation'].value_counts()
    count = count.to_frame()
    per = count.div(count.sum(axis=0))*100
    count['Percentage'] = per['Winner_annotation']
    count['Percentage'] = pd.Series([round(val, 2)
                                     for val in count['Percentage']],
                                    index=count.index)
    count['Percentage'] = pd.Series(["{0:.2f}%".format(val)
                                     for val in count['Percentage']],
                                    index=count.index)

    print("\nAggregated Class Distribution (after majority): "
          "\nTotal: %s \n%s"% (df['Winner_annotation'].shape[0], count))

    conf = df['Winner_annotation'].value_counts()
    conf.plot.bar()
    plt.xlabel('Classes')
    plt.ylabel('Number')
    plt.title('Class distribution')
    plt.tight_layout()
    after_name = 'plots/class_distr_after' + class_column +'.png'
    plt.savefig(after_name)

    # Average agreement (confidence): average of all
    # confidences with >=2 annotations

    ann_gr_1 = df[df['Number_annotations'] == 1]
    count = ann_gr_1['Number_annotations'].count()
    print('\n1 annotation:%s %.2f%%' % (count,
                                         numpy.divide(count,
                                                      df['Number_annotations'].
                                                      sum())*100))

    ann_gr_2 = df[df['Number_annotations'] == 2]
    count = ann_gr_2['Number_annotations'].count()
    print('2 annotations:%s %.2f%%' % (count,
                                       numpy.divide(count,
                                                    df['Number_annotations'].
                                                    sum())*100))

    ann_gr_3 = df[df['Number_annotations'] == 3]
    count = ann_gr_3['Number_annotations'].count()
    print('3 annotations:%s %.2f%%' % (count,
                                       numpy.divide(count,
                                                    df['Number_annotations'].
                                                    sum())*100))

    ann_gr_4 = df[df['Number_annotations'] == 4]
    count = ann_gr_4['Number_annotations'].count()
    print('4 annotations:%s %.2f%%' % (count,
                                       numpy.divide(count,
                                                    df['Number_annotations'].
                                                    sum())*100))
    ann_gr_5 = df[df['Number_annotations'] == 5]
    count = ann_gr_5['Number_annotations'].count()
    print('5 annotations:%s %.2f%%' % (count,
                                       numpy.divide(count,
                                                    df['Number_annotations'].
                                                    sum()) * 100))
    ann_gr_6 = df[df['Number_annotations'] == 6]
    count = ann_gr_6['Number_annotations'].count()
    print('6 annotations:%s %.2f%%' % (count,
                                       numpy.divide(count,
                                                    df['Number_annotations'].
                                                    sum()) * 100))

    ann_gr = df[df['Number_annotations'] >= annotators]

    name = 'aggregated_' + class_column + '.csv'
    ann_gr.to_csv(name, index=False)

    print("\nAverage agreement : %.2f%%" % ann_gr['Confidence'].mean())
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--class_number",type=int, required=True,
                        help="choose one of the three classes: 1 (expressive) ,2 (easy to follow),3 (enjoy)")
    args = parser.parse_args()
    report_annotations('annotations_database.txt',args.class_number,3)