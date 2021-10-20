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

def aggregate_annotations(file, class_number, type, gender, mean_low, mean_high, deviation=0.75, exclude_annotator=None):

    data = pd.read_csv(file)
    
    if gender == 0:
        data = data[data['Sample_name'].str.contains("_male")]
    elif gender == 1:
        data = data[data['Sample_name'].str.contains("_female")]

    if exclude_annotator != None:
        for annotator in exclude_annotator:
            data = data[data.Username != annotator]
    #data = data[data.Username != 'apetrogianni@uth.gr']
    #data = data[data.Username != 'apetrogianni@uth.gr']
    #data = data[data.Username != 'Despina']
    #data = data[data.Username != 'kb@gmail.com']
    #data = data[data.Username != 'electrasif']
    #data = data[data.Username != 'rodo.kasidiari@gmail.com']
    if type == 0:
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
        res=res.drop(['Sample_name'], axis=1)
        res['Max'] = res.idxmax(axis=1)
        res['max_value'] = res.max(axis=1)
        df['Winner_annotation'] = res["Max"]
        df['Confidence'] = res["max_value"]

    else:
        if class_number == 1:
            class_name = 'Class1'
        elif class_number == 2:
            class_name = 'Class2'
        else:
            class_name = 'Class3'

        number_of_annotations, sample_names, mean, dev, winner = [], [], [], [], []
        winner_without_dev = []
        for sample in set(data.Sample_name):
            sample_dataframe = data.loc[data['Sample_name'] == sample]
            number_of_annotations.append(sample_dataframe.shape[0])
            mean_value = sample_dataframe[class_name].mean()
            dev_value  = sample_dataframe[class_name].mad()
            sample_names.append(sample)
            mean.append(mean_value)
            dev.append(dev_value)
            if mean_value <= mean_low:
                winner_without_dev.append("negative")
            elif mean_value >= mean_high:
                winner_without_dev.append("positive")
            else:
                winner_without_dev.append("Nan")

            if mean_value <= mean_low and dev_value < deviation:
                winner.append("negative")
                if len(sample_dataframe[class_name].values)>=3:
                    print(sample_dataframe[class_name].values," mean:",mean_value," σ:", dev_value, " class: negative")
            elif mean_value >= mean_high and dev_value < deviation:
                winner.append("positive")
                if len(sample_dataframe[class_name].values)>=3:
                    print(sample_dataframe[class_name].values," mean:",mean_value," σ:", dev_value, " class: positive")
            else:
                winner.append("Nan")

        # Create Dataframe
        aggregation = {'Sample_Name': [],
                       'Winner_annotation':[],
                       'Winner_without_dev':[],
                       'Mean': [],
                       'Deviation': [],
                       'Number_annotations': []}

        df = pd.DataFrame(aggregation, columns=list(aggregation.keys()))
        df['Sample_Name'] = sample_names
        df['Winner_annotation'] = winner
        df['Winner_without_dev'] = winner_without_dev
        df['Mean'] = mean
        df['Deviation'] = dev
        df['Number_annotations'] = number_of_annotations

    return df



def save_to_csv(df,name):   

    df.to_csv(name, index=False)


def report_annotations(file, class_number,annotators,type,gender,mean_low,mean_high,deviation=0.75,exclude_annotator=None):

    data = pd.read_csv(file)
    
    if gender == 0:
        g = "male"
        print("-->Only male samples")
        data = data[data['Sample_name'].str.contains("_male")]
    elif gender == 1:
        g = "female"
        print("-->Only female samples")
        data = data[data['Sample_name'].str.contains("_female")]

    if exclude_annotator != None:
        for annotator in exclude_annotator:
            data = data[data.Username != annotator]
    #data = data[data.Username != 'apetrogianni@uth.gr']
    #data = data[data.Username != 'Despina']
    #data = data[data.Username != 'kb@gmail.com']
    #data = data[data.Username != 'electrasif']
    #data = data[data.Username != 'rodo.kasidiari@gmail.com']
    df = aggregate_annotations(file,class_number,type,gender,mean_low,mean_high,deviation,exclude_annotator)
    csv_name = 'aggregate_annotations_of_class' + str(class_number) + g +'.csv'
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
              startangle=90, figsize=(30, 20), autopct='%1.1f%%')
    pie_name = 'plots/pie' + class_column + '.png'
    plt.savefig(pie_name)
    plt.close()

    # Class distribution (before majority or averaging) + plot

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


    print("\nInitial Class Distribution (before majority or averaging): "
            "\nTotal: %s \n%s" % (data[class_column].shape[0], count))



    conf = data[class_column].value_counts()
    conf.plot.bar()
    plt.xlabel('Classes')
    plt.ylabel('Number')
    plt.title('Class distribution')
    plt.tight_layout()
    before_name = 'plots/class_distr_before' + class_column + g +'.png'
    plt.savefig(before_name)
    plt.close()

    # Class distribution (after majority or averaging) +  without deviation (σ)
    count = df['Winner_without_dev'].value_counts()
    count = count.to_frame()
    per = count.div(count.sum(axis=0)) * 100
    count['Percentage'] = per['Winner_without_dev']
    count['Percentage'] = pd.Series([round(val, 2)
                                     for val in count['Percentage']],
                                    index=count.index)
    count['Percentage'] = pd.Series(["{0:.2f}%".format(val)
                                     for val in count['Percentage']],
                                    index=count.index)

    print(
        "\nAggregated Class Distribution (after majority or averaging), before applying minimum number of annotators: "
        "\nBefore applying deviation limitation (σ) \nTotal: %s \n%s" % (df['Winner_without_dev'].shape[0], count))
    #plot
    conf = df['Winner_without_dev'].value_counts()
    conf.plot.bar()
    plt.xlabel('Classes')
    plt.ylabel('Number')
    plt.title('Class distribution')
    plt.tight_layout()
    after_name = 'plots/class_distr_mean' + class_column + g + '.png'
    plt.savefig(after_name)
    plt.close()

    # Class distribution (after majority or averaging) + plot with deviation (σ)
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

    print("\nAfter applying deviation limitation (σ) \nTotal: %s \n%s"% (df['Winner_annotation'].shape[0], count))

    conf = df['Winner_annotation'].value_counts()
    conf.plot.bar()
    plt.xlabel('Classes')
    plt.ylabel('Number')
    plt.title('Class distribution')
    plt.tight_layout()
    after_name = 'plots/class_distr_dev' + class_column + g +'.png'
    plt.savefig(after_name)
    plt.close()

    # Average agreement (confidence): average of all
    # confidences with >=2 annotations

    ann_gr_1 = df[df['Number_annotations'] == 1]
    count = ann_gr_1['Number_annotations'].count()
    print('\n1 annotation:%s %.2f%%' % (count,
                                         numpy.divide(count,
                                                      df['Winner_annotation'].shape[0])*100))

    ann_gr_2 = df[df['Number_annotations'] == 2]
    count = ann_gr_2['Number_annotations'].count()
    print('2 annotations:%s %.2f%%' % (count,
                                       numpy.divide(count,
                                                    df['Winner_annotation'].shape[0])*100))

    ann_gr_3 = df[df['Number_annotations'] == 3]
    count = ann_gr_3['Number_annotations'].count()
    print('3 annotations:%s %.2f%%' % (count,
                                       numpy.divide(count,
                                                    df['Winner_annotation'].shape[0])*100))

    ann_gr_4 = df[df['Number_annotations'] == 4]
    count = ann_gr_4['Number_annotations'].count()
    print('4 annotations:%s %.2f%%' % (count,
                                       numpy.divide(count,
                                                    df['Winner_annotation'].shape[0])*100))
    ann_gr_5 = df[df['Number_annotations'] == 5]
    count = ann_gr_5['Number_annotations'].count()
    print('5 annotations:%s %.2f%%' % (count,
                                       numpy.divide(count,
                                                    df['Winner_annotation'].shape[0]) * 100))
    ann_gr_6 = df[df['Number_annotations'] == 6]
    count = ann_gr_6['Number_annotations'].count()
    print('6 annotations:%s %.2f%%' % (count,
                                       numpy.divide(count,
                                                    df['Winner_annotation'].shape[0]) * 100))
    ann_gr_7 = df[df['Number_annotations'] == 7]
    count = ann_gr_7['Number_annotations'].count()
    print('7 annotations:%s %.2f%%' % (count,
                                       numpy.divide(count,
                                                    df['Winner_annotation'].shape[0]) * 100))
    ann_gr_8 = df[df['Number_annotations'] == 8]
    count = ann_gr_8['Number_annotations'].count()
    print('8 annotations:%s %.2f%%' % (count,
                                       numpy.divide(count,
                                                    df['Winner_annotation'].shape[0]) * 100))
    ann_gr_9 = df[df['Number_annotations'] == 9]
    count = ann_gr_9['Number_annotations'].count()
    print('9 annotations:%s %.2f%%' % (count,
                                       numpy.divide(count,
                                                    df['Winner_annotation'].shape[0]) * 100))
    ann_gr_10 = df[df['Number_annotations'] == 10]
    count = ann_gr_10['Number_annotations'].count()
    print('10 annotations:%s %.2f%%' % (count,
                                       numpy.divide(count,
                                                    df['Winner_annotation'].shape[0]) * 100))
    if type == 0:
        ann_gr = df[df['Number_annotations'] >= annotators]

        name = 'aggregated_' + class_column + '.csv'
        ann_gr.to_csv(name, index=False)

        print("\nAverage agreement : %.2f%%" % ann_gr['Confidence'].mean())
        print("\n")
    else:
        ann_gr = df[df['Number_annotations'] >= annotators]

        ##class distribution after applying minimum annotators
        count = ann_gr['Winner_without_dev'].value_counts()
        count = count.to_frame()
        per = count.div(count.sum(axis=0)) * 100
        count['Percentage'] = per['Winner_without_dev']
        count['Percentage'] = pd.Series([round(val, 2)
                                         for val in count['Percentage']],
                                        index=count.index)
        count['Percentage'] = pd.Series(["{0:.2f}%".format(val)
                                         for val in count['Percentage']],
                                        index=count.index)

        print(
            "\nAggregated Class Distribution (after majority or averaging), after applying minimum number of annotators: "
            "\nBefore applying deviation limitation (σ) \nTotal: %s \n%s" % (ann_gr['Winner_without_dev'].shape[0], count))

        ##class distribution after applying minimum annotators
        count = ann_gr['Winner_annotation'].value_counts()
        count = count.to_frame()
        per = count.div(count.sum(axis=0)) * 100
        count['Percentage'] = per['Winner_annotation']
        count['Percentage'] = pd.Series([round(val, 2)
                                         for val in count['Percentage']],
                                        index=count.index)
        count['Percentage'] = pd.Series(["{0:.2f}%".format(val)
                                         for val in count['Percentage']],
                                        index=count.index)

        print("\nAfter applying deviation limitation (σ) \nTotal: %s \n%s" % (ann_gr['Winner_annotation'].shape[0], count))

        #plot
        conf = ann_gr['Winner_annotation'].value_counts()
        conf.plot.bar()
        plt.xlabel('Classes')
        plt.ylabel('Number')
        plt.title('Class distribution')
        plt.tight_layout()
        after_name = 'plots/class_distr_annotators' + class_column + g + '.png'
        plt.savefig(after_name)
        plt.close()

        print("\nAverage disagreement (average σ) : %.2f" % ann_gr['Deviation'].mean())
        ann_grB = ann_gr[ann_gr['Winner_annotation']!='Nan']
        print("\nTotal number of final samples : %.0f" % ann_grB['Winner_annotation'].shape[0])
        name = 'aggregated_' + class_column + g +'.csv'
        ann_grB.to_csv(name, index=False)

        ####compute average distance between user and mean
        if class_number == 1:
            class_name = 'Class1'
        elif class_number == 2:
            class_name = 'Class2'
        else:
            class_name = 'Class3'
        users = []
        mean_distances = []
        num_of_samples = []
        data = data[data.Sample_name.isin(ann_gr.Sample_Name)]
        for user in set(data.Username):
            sample_dataframe = data.loc[data['Username'] == user]
            number_of_samples = sample_dataframe.shape[0]
            total_distance = 0
            for index, row in sample_dataframe.iterrows():
                mean_of_sample = df.loc[df['Sample_Name'] == row['Sample_name']]
                mean_of_sample = mean_of_sample['Mean'].item()
                total_distance += abs(mean_of_sample - row[class_name])
            users.append(user)
            mean_distances.append(total_distance / number_of_samples)
            num_of_samples.append(number_of_samples)

        print("\nAverage distance between each annotator and mean value:\n")
        for count, user in enumerate(users):
            print("User:", user,"\nAverage Distance:",mean_distances[count],"\nNumber of samples annotated by user:",num_of_samples[count])
            print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--class_number",type=int, required=True,
                        help="choose one of the three classes: 1 (expressive) ,2 (easy to follow),3 (enjoy)")
    parser.add_argument("-a", "--annotators", type=int, required=True,
                        help="minimum number of annotators to take into account")
    parser.add_argument("-t","--type_of_aggregation", type=int, required=True,
                        help="Choose between majority vote (0) or averaging (1)")
    parser.add_argument("-g", "--gender", type=int, required=True,
                        help="Choose between male (0) or female (1)")
    parser.add_argument("-ml", "--mean_low", type=float, required=True,
                        help="Low mean threshold")
    parser.add_argument("-mh", "--mean_high", type=float, required=True,
                        help="High mean threshold")
    parser.add_argument("-d", "--deviation", type=float, required=False, default=0.75,
                        help="deviation threshold")
    parser.add_argument("-ea", "--exclude_annotator", type=str, nargs='+', required=False, default=None, help="the annotators' names to be excluded the from the proceedings")
    args = parser.parse_args()
    report_annotations('annotations_database.txt',args.class_number,args.annotators,args.type_of_aggregation,args.gender,args.mean_low,args.mean_high,args.deviation,args.exclude_annotator)
