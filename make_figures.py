import pandas as pd
import plotly.graph_objects as go
import numpy as np
from text_analysis import text_based_feature_extraction
import json

from audio_analysis import audio_based_feature_extraction


def load_conf_file(path):
    with open(path) as f:
        conf = json.load(f)
    return conf

def make_figures():
    conf = load_conf_file('config.json')
    input_file=conf['audiofile']
    google_credentials=conf['google_credentials']
    reference_text = conf['reference_text']

    #text feature extraction
    text_features,text_feature_names,text_metadata = text_based_feature_extraction(input_file,google_credentials,reference_text)
    #audio feature extraction
    audio_features, audio_feature_names, audio_metadata = audio_based_feature_extraction(input_file)
    '''
    rec = text_features[0]
    pre = text_features[1]
    f1 = text_features[2]
    words = text_features[3]
    dur = text_features[4]
    word_rate = text_features[5]
    '''

    # take x and y coordinates for temporal scores plotting
    recall_list = text_metadata["temporal_recall"]
    precision_list  = text_metadata["temporal_precision"]
    f1_list = text_metadata["temporal_f1"]
    Ref = text_metadata["temporal_ref"]
    Asr = text_metadata["temporal_asr"]
    X_recall=[]
    Y_recall=[]
    X_precision=[]
    Y_precision=[]
    X_f1=[]
    Y_f1=[]
    for i in range(len(recall_list)):
        X_recall.append(recall_list[i]['x'])
        Y_recall.append(recall_list[i]['y'])
        X_precision.append(precision_list[i]['x'])
        Y_precision.append(precision_list[i]['y'])
        X_f1.append(f1_list[i]['x'])
        Y_f1.append(f1_list[i]['y'])

    X_Rlist = np.array(X_recall)
    Y_Rlist = np.array(Y_recall)
    X_Plist = np.array(X_precision)
    Y_Plist = np.array(Y_precision)
    X_Flist = np.array(X_f1)
    Y_Flist = np.array(Y_f1)

    '''
    # audio features extraction
    
    average_silence_dur_short = audio_features[0]
    average_silence_dur_long = audio_features[1]
    silence_seg_per_minute_short = audio_features[2]
    silence_seg_per_minute_long = audio_features[3]
    std_short= audio_features[4]
    std_long = audio_features[5]
    high_percentage = audio_features[6]
    neutral_percentage = audio_features[7]
    low_percentage = audio_features[8]
    '''
    text_metadata_new = {i:text_metadata[i] for i in text_metadata if i!='Asr timestamps' and i!='temporal_recall' and i!='temporal_precision' and i!='temporal_f1' and i!='temporal_ref' and i!='temporal_asr'}
    text_metadata_names =list(text_metadata_new.keys())
    text_metadata_values = list(text_metadata_new.values())
    Text_names = text_feature_names + text_metadata_names
    audio_metadata_names =list(audio_metadata.keys())
    audio_metadata_values = list(audio_metadata.values())
    Audio_names = audio_feature_names + audio_metadata_names
    Names = Text_names + Audio_names
    Values = text_features + text_metadata_values + audio_features + audio_metadata_values
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
        cells=dict(values=[Names, Values],
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
