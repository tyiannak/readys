# Agreement of Annotations 
The purpose of this code is to aggregate the annotations in a manner of agreement between the annotators, so that we can come up with one label per sample and produce some statistics about the agreement of annotations.

### annotations_database.txt 
This file contains the annotations as selected by users on the UI. Each annotation occupies a line with elements : Timestamp,Sample_name,Class1,Class2,Class3 that are seperated by a comma. The elements Class1,Class2 and Class3 refer to the three different classification tasks. 

An example of such a file is given. 

### audiofiles.txt 
This file contains the names of all the audio files that exists in our dataset, seperated by line. 

An example of such a file is given.  

### aggregate_annotations.py 
In order to run the aggregation of annotations write the following command: 

```
python3 aggregate_annotations.py -c class_number -a annotators -t type_of_aggregation -g gender -ml mean_low -mh mean_high -d deviation
``` 

Where: 

- class_number: an integer (1,2 or 3) that defines the classification task we want to aggregate the annotations for. 

- annotators: an integer that defines the annotator's threshold. For example, if annotators = 3 then we take into account only the samples that are annotated by 3 or more annotators. 
- type_of_aggregation: use 0 for majority vote or 1 for averaging
- gender: use 0 for male samples or 1 for female samples
- mean_low: the threshold below which the mean value of the annotations must be obeyed in order for the sample to be considered as negative
- mean_high: the threshold above which the mean value of the annotations must be obeyed in order for the sample to be considered as positive
- deviation: the threshold below which the deviation of the annotations must be obeyed in order for the sample to be valid (not to be discarded)

### results
When running the above command, the following files will be produced: 

- aggergate_annotations_of_classXgender.csv : In case of majority vote, this file contains the winner annotation, the confidence of the agreement and the number of annotations per sample (for all samples regardless of the number of annotations). These concern the classification task X (where X = 1,2 or 3). In case of averaging, this file contains the winner annotation produced by mean and deviation limitatons, the winner_without_dev produced by mean limitation, the mean, the deviation and the number of annotations per sample. 

- aggregated_ClassXgender.csv : this file contains the same information as aggregate_annotations_of_classX.csv, but only for the samples with number_of_annotations >= annotators (as specified in the input)  

- plots/pieClassX.png : a pie figure with the distribution of annotations among users for classification task X.  

- plots/class_distr_beforeClassXgender.png :  a figure of number of annotatinos per class for classification task X (taking into account all the inital annotations, before averaging or majority voting) 

- plots/class_distr_meanClassXgender.png : a figure of number of annotations per class for classification task X after applying mean thresholding
- plots/class_distr_devClassXgender.png : a figure of number of annotations per class for classification task X after applying deviation thresholding
- plots/class_distr_annotatorsClassXgender.png : a figure of number of annotations per class for classification task X after applying minimum number of annotators

In addition to the aforementioned generated files, some statistics will be printed in the command line.
An example of these results can be found here: https://docs.google.com/document/d/1KIPBDB2i6NJzQsRfwPmQAiHGYrPSUDPeLyEDy3Cqqiw/edit?usp=sharing

### recording_level_dataset_parser.py 
This file is responsible for producing dataset in the form of subfolders per class, after the aggregation of annotations has been completed. 

To run this file: 
```
python3 recording_level_dataset_parser.py -n name_of_classification_task -aa aggregate_annotations -i input_data
``` 
Where: 

- name_of_classification_task: the name of the classification task e.g. expressive/enjoyable etc 
- aggregate_annotations: the csv file which contains the winner annotations per sample (aggregated_ClassX.csv which has been produced by the aggregate_annotations.py) 
- input_data: the directory with all audio files collected 

The above command will create a directory datasets/name_of_classification_task which will contain all samples from aggregated_ClassX.csv organized in subfolders.

### convert.py 
This file is responsible for converting all the audio files into wav format, with defined sampling rate and channels. It also replaces the whitespaces in the file name with underscores. 

To run this file:
```
python3 convert.py files_folder samplingRate channels out_path
```
Where: 

- files_folder: the folder of the audio files to be converted
- samplingRate: the sampling rate into which audio files will be converted
- channels: the number of channels to use
- out_path: the directory for the converted files to be stored
