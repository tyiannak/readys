from csv import reader
import pandas as pd
import plotly.graph_objects as go

from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioBasicIO as aio

def make_figures():

    #read data that main produced
    df = pd.read_pickle("score.pkl")
    alignment = df.loc[0,'alignment']
    print(alignment)
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
    seg_limits = aS.silence_removal(x, fs, 0.006, 0.006,0.07)
    print(seg_limits)
    total_speech = 0.0
    for k in seg_limits:
        word_speech = k[1] - k[0]
        total_speech = total_speech + word_speech
    speech_ratio=total_speech/dur
    total_silence=dur-total_speech
    number_of_pauses=len(seg_limits)

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


    fig = go.Figure(data=[go.Table(
        columnorder=[1, 2],
        columnwidth=[80, 400],
        header=dict(values=[[],['<b>SCORES</b>']],
                    line_color='darkslategray',
                    fill_color='royalblue',
                    align='center',
                    font=dict(color='white', size=12),
                    height=40
        ),
        cells=dict(values=[["Recall Score (%)","Precision Score (%)","F1 Score (%)","Number of words","Total Duration (sec)","Word Rate (words/min)","Speech Ratio (sec)","Word Rate in Speech (words/sec)","Total Silence Duration (sec)","Number of Pauses"],
                           [float("{:.2f}".format(rec)), float("{:.2f}".format(pre)), float("{:.2f}".format(f1)),words,dur,float("{:.2f}".format(words/(dur/60.0))),float("{:.2f}".format(speech_ratio)),float("{:.2f}".format(words/total_speech)),float("{:.2f}".format(total_silence)),number_of_pauses]],
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
