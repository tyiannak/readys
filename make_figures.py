from csv import reader
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioBasicIO as aio
from pyAudioAnalysis.audioTrainTest import file_classification
from segment_level_classifier import segment_classification

def make_figures():

    #read data that main produced
    df = pd.read_pickle("score.pkl")
    alignment = df.loc[0,'alignment']
    #print(alignment)
    rec = df.loc[0, 'recall']
    pre = df.loc[0, 'precision']
    f1 = df.loc[0, 'f1']
    words = df.loc[0, 'words']
    dur = df.loc[0, 'dur']
    df1 = pd.read_pickle("recall_temporal.pkl")
    df2 = pd.read_pickle("precision_temporal.pkl")
    df3 = pd.read_pickle("f1_temporal.pkl")

    #audio features
    fs, x = aio.read_audio_file("output.wav")
    seg_limits_short = aS.silence_removal(x, fs, 0.06, 0.06,0.07)
    #print(seg_limits_short)
    seg_limits_long = aS.silence_removal(x,fs,0.1,0.1,0.3)
    #print(seg_limits_long)

    total_speech_short = 0.0
    silence_durations_short = []
    counter=0
    for k in seg_limits_short:
        if counter==0:
            silence_durations_short.append(k[0])
        elif counter==(len(seg_limits_short)-1):
            p = counter - 1
            silence_durations_short.append(k[0] - seg_limits_short[p][1])
            silence_durations_short.append(dur-k[1])
        else:
            p=counter-1
            silence_durations_short.append(k[0]-seg_limits_short[p][1])
        word_speech_short = k[1] - k[0]
        total_speech_short = total_speech_short + word_speech_short
        counter=counter+1
    speech_ratio_short=total_speech_short/dur
    number_of_pauses_short=len(silence_durations_short)
    #print(silence_durations_short)

    total_speech_long = 0.0
    counter=0
    silence_durations_long = []
    for k in seg_limits_long:
        if counter==0:
            silence_durations_long.append(k[0])
        elif counter==(len(seg_limits_long)-1):
            p = counter - 1
            silence_durations_long.append(k[0] - seg_limits_long[p][1])
            silence_durations_long.append(dur-k[1])
        else:
            p=counter-1
            silence_durations_long.append(k[0]-seg_limits_long[p][1])
        word_speech_long = k[1] - k[0]
        total_speech_long = total_speech_long + word_speech_long
        counter=counter+1
    #print(seg_limits_short)
    #print(silence_durations_short)
    #print(seg_limits_long)
    #print(silence_durations_long)
    speech_ratio_long = total_speech_long / dur
    number_of_pauses_long = len(silence_durations_long)
    silence_durations_short = np.array(silence_durations_short)
    silence_durations_long = np.array(silence_durations_long)
    std_short = np.std(silence_durations_short)
    std_long = np.std(silence_durations_long)
    average_silence_dur_short= np.mean(silence_durations_short)
    average_silence_dur_long = np.mean(silence_durations_long)

    #take x and y coordinates
    X_Rlist = df1['x'].to_numpy()
    Y_Rlist = df1['y'].to_numpy()
    X_Plist = df2['x'].to_numpy()
    Y_Plist = df2['y'].to_numpy()
    X_Flist = df3['x'].to_numpy()
    Y_Flist = df3['y'].to_numpy()

    #this is for temporal  alignment(mouseover)
    with open('Ref.csv','r') as read_obj:
        csv_reader = reader(read_obj)
        Ref=list(csv_reader)
    with open('Asr.csv','r') as read_obj:
        csv_reader = reader(read_obj)
        Asr=list(csv_reader)

    #classify the whole recording between low,neutral and high
    input_file = "output.wav"
    model_name = "segment_classifier"
    model_type = "svm_rbf"
    class_id, probability, classes = file_classification(input_file, model_name, model_type)

    if class_id == 0.0:
        wav_class = "High"
    elif class_id == 1.0:
        wav_class = "Neutral"
    else:
        wav_class = "Low"

    # statistic segment classification
    classes, high_percentage, neutral_percentage, low_percentage = segment_classification(input_file,model_name)

    #create feature vector for the whole recording
    DF=pd.DataFrame(
        {
            "average silence duration" : [float("{:.2f}".format(average_silence_dur_short)),float("{:.2f}".format(average_silence_dur_long))],
            "silence_seg_per_minute" : [float("{:.2f}".format(number_of_pauses_short/(dur/60.0))),float("{:.2f}".format(number_of_pauses_long/(dur/60.0)))],
            "std" : [float("{:.2f}".format(std_short)),float("{:.2f}".format(std_long))]
        }
    )
    DF.to_csv('whole_recording_feature_vector.csv',index=False)
    fig = go.Figure(data=[go.Table(
        columnorder=[1, 2],
        columnwidth=[80, 400],
        header=dict(values=[['<b>SCORES</b>']],
                    line_color='darkslategray',
                    fill_color='royalblue',
                    align='center',
                    font=dict(color='white', size=12),
                    height=40
        ),
        cells=dict(values=[["Recall Score (%)","Precision Score (%)","F1 Score (%)","Number of words","Total Duration (sec)","Word Rate (words/min)","Speech Ratio (sec) - Short windows","Speech Ratio (sec) - Long windows","Word Rate in Speech (words/sec) - Short windows","Word Rate in Speech (words/sec) - Long windows","Silence Average Duration (sec) - Short windows","Silence Average Duration (sec) - Long windows","Number of Pauses - Short windows","Number of Pauses - Long windows","Standard deviation of silence duration (sec) - Short windows","Standard deviation of silence duration (sec) - Long windows","Class of whole recorded file","High class (%)","Neutral class (%)","Low class(%)"],
                           [float("{:.2f}".format(rec)), float("{:.2f}".format(pre)), float("{:.2f}".format(f1)),words,dur,float("{:.2f}".format(words/(dur/60.0))),float("{:.2f}".format(speech_ratio_short)),float("{:.2f}".format(speech_ratio_long)),float("{:.2f}".format(words/total_speech_short)),float("{:.2f}".format(words/total_speech_long)),float("{:.2f}".format(average_silence_dur_short)),float("{:.2f}".format(average_silence_dur_long)),number_of_pauses_short,number_of_pauses_long,float("{:.2f}".format(std_short)),float("{:.2f}".format(std_long)),wav_class,float("{:.2f}".format(high_percentage)),float("{:.2f}".format(neutral_percentage)),float("{:.2f}".format(low_percentage))]],
                   line_color='darkslategray',
                   fill=dict(color=['paleturquoise', 'white']),
                   align=['left', 'center'],
                   font_size=12,
                   height=30)
        )
    ])
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=X_Rlist, y=Y_Rlist,
                              mode='lines+markers',
                              name='recall score',
                              marker=dict(color='rgb(179,226,205)')))

    fig1.add_trace(go.Scatter(x=X_Plist, y=Y_Plist,
                              mode='lines+markers',
                              name='precision score',
                              marker=dict(color='rgb(253,205,172)')))
    fig1.add_trace(go.Scatter(x=X_Flist, y=Y_Flist,
                              mode='lines+markers',
                              name='F1 score',
                              marker=dict(color='rgb(127,60,141)'),
                              text=['Reference Text :{} \n Asr Text:{}'.format(Ref[i],Asr[i]) for i in range(len(X_Flist))],
                              #text=['Asr Text {}'.format(Asr[i]) for i in range(len(X_Flist))],
                              hovertemplate=
                              '<b>%{text}</b>',
                              showlegend=True
                              ))

    fig1.update_layout(
        #autosize=False,
        #width=1500,
        height=600,
        yaxis=dict(
            title_text="Percentage temporal score",
            tickmode="array",
            titlefont=dict(size=30),
        ),
        xaxis=dict(
            title_text="Center of window(time)",
        )
    )

    return fig, fig1
