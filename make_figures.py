import plotly.graph_objects as go
import numpy as np
from text_analysis import get_asr_features as tbfe
import json
from audio_analysis import audio_based_feature_extraction as abfe
from models.utils import load_classifiers
from recording_level_analysis import get_final_class

def load_conf_file(path):
    with open(path) as f:
        conf = json.load(f)
    return conf


def make_figures():
    conf = load_conf_file('config.json')
    input_file=conf['audiofile']
    google_credentials=conf['google_credentials']
    reference_text = conf['reference_text']
    audio_models_directory = conf['audio_models_directory']
    text_models_directory = conf['text_models_directory']
    recording_level_model_directory = conf['recording_level_model_directory']
    segmentation_threshold = conf['segmentation_threshold']
    segmentation_method = conf['segmentation_method']
    classifiers_attributes = load_classifiers(text_models_directory)

    #text feature extraction
    text_features, text_feature_names, text_metadata = tbfe(input_file,
                                                            google_credentials,
                                                            classifiers_attributes,
                                                            reference_text,
                                                            segmentation_threshold,
                                                            segmentation_method)
    # audio feature extraction
    if text_metadata['Number of words']== 0:
        audio_feature_names = ["Average silence duration short (sec)",
                               "Average silence duration long (sec)",
                               "Silence segments per min - short (segs/min)",
                               "Silence segments per min - long (segs/min)",
                               "Std short",
                               "Std long",
                               "Speech ratio short (sec)",
                               "Speech ratio long (sec)",
                               "Word rate in speech short (words/sec)",
                               "Word rate in speech long (words/sec)"]
        audio_features = [0] * len(audio_feature_names)
        audio_metadata = {
            "Number of pauses short": 0,
            "Number of pauses long": 0,
            "Total speech duration short (sec)": 0,
            "Total speech duration long (sec)": 0
        }
    else:
        audio_features, audio_feature_names, audio_metadata = abfe(input_file,audio_models_directory)

    #recording level final class
    final_class , category = get_final_class(input_file,recording_level_model_directory)

    # take x and y coordinates for temporal text scores plotting
    X_recall = []
    Y_recall = []
    X_precision = []
    Y_precision = []
    X_f1 = []
    Y_f1 = []
    if reference_text:
        recall_list = text_metadata["temporal_recall"]
        precision_list  = text_metadata["temporal_precision"]
        f1_list = text_metadata["temporal_f1"]
        Ref = text_metadata["temporal_ref"]
        Asr = text_metadata["temporal_asr"]

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

    text_metadata_new = {i:text_metadata[i]
                         for i in text_metadata
                         if i!='asr timestamps' and i!='temporal_recall'
                         and i!='temporal_precision' and i != 'temporal_f1'
                         and i!='temporal_ref' and i != 'temporal_asr'}
    text_metadata_names = list(text_metadata_new.keys())
    text_metadata_values = list(text_metadata_new.values())
    Text_names = text_feature_names + text_metadata_names
    audio_metadata_names = list(audio_metadata.keys())
    audio_metadata_values = list(audio_metadata.values())
    Audio_names = audio_feature_names + audio_metadata_names
    Names = Text_names + Audio_names + [category]
    Values = text_features + text_metadata_values + audio_features + \
             audio_metadata_values + [final_class]
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
                              text=['Reference Text :{} \n '
                                    'Asr Text:{}'.format(Ref[i],Asr[i])
                                    for i in range(len(X_Flist))],
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
