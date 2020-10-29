from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis.audioTrainTest import load_model, classifier_wrapper
from pyAudioAnalysis import MidTermFeatures as aF
import numpy as np

def segment_classification(input_file, model_name,model_type):
    # Load classifier:

    classifier, mean, std, classes, mid_window, mid_step, short_window, \
        short_step, compute_beat = load_model(model_name)

    # read audio file and convert to mono
    sampling_rate, signal = audioBasicIO.read_audio_file(input_file)
    signal = audioBasicIO.stereo_to_mono(signal)

    if sampling_rate == 0:
        # audio file IO problem
        return -1, -1, -1
    if signal.shape[0] / float(sampling_rate) < mid_window:
        mid_window = signal.shape[0] / float(sampling_rate)

    # feature extraction:
    mid_features, s, _ = \
        aF.mid_feature_extraction(signal, sampling_rate,
                                  mid_window * sampling_rate,
                                  mid_step * sampling_rate,
                                  round(sampling_rate * short_window),
                                  round(sampling_rate * short_step))
    classes = []
    #print(mid_features)
    #take every sample (every mid term window) and not every feature
    mid_features=mid_features.tolist()
    #print(mid_features)
    tlist = list(zip(*mid_features))
    #print(tlist)
    tlist = np.array(tlist)
    #print(tlist)
    #print(len(tlist))
    for i in tlist:
        print(i)
        feature_vector = (i - mean) / std  # normalization
        print(feature_vector)
        class_id, probability = classifier_wrapper(classifier,model_type,
                                                   feature_vector)
        classes.append(class_id)
    num_of_highs = 0
    num_of_neutrals = 0
    num_of_lows = 0
    for i in classes:
        if i==0.0:
            num_of_highs = num_of_highs +1
        elif i==1.0:
            num_of_neutrals = num_of_neutrals +1
        else:
            num_of_lows = num_of_lows +1
    high_percentage = num_of_highs * 100/len(classes)
    neutral_percentage = num_of_neutrals * 100/len(classes)
    low_percentage = num_of_lows * 100/len(classes)

    return classes,high_percentage,neutral_percentage,low_percentage
