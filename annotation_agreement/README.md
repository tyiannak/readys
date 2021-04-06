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
python3 aggregate_annotations.py -c class_number -a annotators
``` 

Where: 

- class_number is an integer (1,2 or 3) that defines the classification task we want to aggregate the annotations for. 

- annotators is an integer that defines the annotator's threshold. For example, if annotators = 3 then we take into account only the samples that are annotated by 3 or more annotators. 

### results
When running the above command, the following files will be produced: 

- aggergate_annotations_of_classX.csv : this file contains the winner annotation, the confidence of the agreement and the number of annotations per sample (for all samples regardless of the number of annotations). These concern the classification task X (where X = 1,2 or 3). 

- aggregated_ClassX.csv : this file contains the same information as aggregate_annotations_of_classX.csv, but only for the samples withe number_of_annotations >= annotators (as specified in the input)  

- plots/pieClassX.png : a pie figure with the distribution of annotations among users for classification task X.  

- plots/class_distr_beforeClassX.png :  a figure of number of annotatinos per class for classification task X (taking into account all the inital annotations, before averaging) 

- plots/class_distr_afterClassX.png : a figure of number of annotations per class for classification task X (taking into account the resulting winner_annotations) 

In addition to the aforementioned generated files, some statistics will be printed in the command line.
An example of these results can be found here: https://docs.google.com/document/d/1KIPBDB2i6NJzQsRfwPmQAiHGYrPSUDPeLyEDy3Cqqiw/edit?usp=sharing
