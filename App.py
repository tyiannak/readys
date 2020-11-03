import dash
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_html_components as html

import plotly.graph_objects as go


import json
from make_figures import make_figures

import numpy as np
import pyaudio
import struct


from scipy.io import wavfile

from os import listdir




app = dash.Dash(__name__, suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                )
global mid_buf
mid_buf = []
global RECORD
global block
block = bytes()

filenames = [f[:-4] for f in listdir("texts/")]
filenames2 = [f for f in listdir("texts/")]
def generate_text_button(k,text_name):
    return dbc.Col(html.A(html.Button(text_name, id=text_name, n_clicks=0),href = '/page-' + str(k)))


def record():
    fs = 44100
    FORMAT = pyaudio.paInt16
    block_size = 0.2  # window size in seconds

    # inialize recording process
    mid_buf_size = int(fs * block_size)
    global mid_buf
    global RECORD
    RECORD = True
    mid_buf = []
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=1, rate=fs,
                     input=True, frames_per_buffer=mid_buf_size)

    while (RECORD):
        global block
        block = stream.read(mid_buf_size)
        count_b = len(block) / 2
        format = "%dh" % (count_b)
        shorts = struct.unpack(format, block)
        cur_win = list(shorts)
        mid_buf = mid_buf + cur_win
        # print(mid_buf)
        del cur_win

    pa.close(stream)



url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

layout_index = html.Div(style={'text-align': 'center'}, children=[
    dbc.Row(dbc.Col(html.H1("Select a text for reading"))),
    dbc.Row(
        [
            #html.Div(children=(generate_text_button(k,i) for k,i in enumerate(filenames)))
            generate_text_button(k,i) for k,i in enumerate(filenames)
            #dbc.Col(children=(generate_text_button(k,i) for k,i in enumerate(filenames)),style={"margin-left": "15px"})
        ]
    ),
])

layout_page_1 = html.Div([
    html.Div(
        style=
        {'height': '70px',
         'border': '5px outset red',
         'background-color': '#ABBAEA',
         'text-align': 'center',
         'font-size': '20px'},
        children=
        [dcc.Markdown(id='text'), ]
    ),
    html.Div(style={'text-align': 'center','display':'none'}, id='graph-container', children=[
        dcc.Graph(id='live-update-graph'),
    ]),
    dcc.Interval(
        id='interval-component',
        interval=200,  # in milliseconds
        n_intervals=0,
        disabled=True,
    ),
    html.Div(style={'text-align': 'center'}, children=[
        html.Button('Record', id='btn', n_clicks=0),
        dcc.Link(html.Button('Stop', id='btn2', n_clicks=0,disabled=True), href='/score'),
        html.Div(id='output', children='Hit the button to update'),
        # dcc.Link('Show score',href='/score'),
    ]),
])

layout_page_2 = html.Div([
    html.Div(
        id='display-text',
        style=
        {'text-align': 'center',
         'font-size': '20px'},
        children=
        [dcc.Markdown(id='text2', children="Please wait for your score to be computed..."),
         dbc.Spinner(color="warning"),
        ]
    ),
    html.Div(style={'display': 'none'}, id='graphs', children=[
        dcc.Graph(
            id='graph1'
        ),
        dcc.Graph(
            id='graph2'),
    ]),
])

# index layout
app.layout = url_bar_and_content_div

# "complete" layout
app.validation_layout = html.Div([
    url_bar_and_content_div,
    layout_index,
    layout_page_1,
    layout_page_2,
])


# Index callbacks
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == "/":
        return layout_index
    elif pathname == "/score":
        return layout_page_2
    else:
        return layout_page_1


# Page 1 callbacks
@app.callback(
    Output('text', 'children'),
    [Input('url', 'pathname')]
)
def text(pathname):
    text_markdown = "\t"
    for i in range(len(filenames2)):
        if pathname == ("/page-" + str(i)):
            with open("texts/"+ filenames2[i]) as this_file:
                for a in this_file.read():
                    if "\n" in a:
                        text_markdown += "\n \t"
                    else:
                        text_markdown += a
            with open('config.json', 'r') as file:
                json_data = json.load(file)
                json_data['reference_text'] = "texts/" + filenames2[i]
            with open('config.json', 'w') as file:
                json.dump(json_data, file)
    return text_markdown

#Page 2 callbacks
#start recording when record is pressed and stop it when stop is pressed
@app.callback(
    Output('btn2','disabled'),
    [Input('btn','n_clicks')],
    prevent_initial_call=True
)
def disable(n):
    return False

@app.callback(
    Output('output', 'children'),
    [Input('btn', 'n_clicks'), Input('btn2', 'n_clicks')],
    prevent_initial_call=True

)
def update(n1, n2):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    global mid_buf
    global RECORD
    # if record button is pressed, start recording(RECORD=True)
    if (button_id == 'btn'):
        RECORD = True
        record()
        return 'Stopped'
    # if stop button is pressed,stop recording (RECORD=False) and write wav file
    elif (button_id == 'btn2'):
        RECORD = False
        fs = 44100
        wavfile.write("output.wav", fs, np.int16(mid_buf))
        return 'Finished'


#enable time intervals when record is pressed
@app.callback(
    Output('interval-component','disabled'),
    [Input('btn', 'n_clicks'), Input('btn2', 'n_clicks')],
    prevent_initial_call=True
)
def time_enable(n1,n2):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if (button_id == 'btn'):
        return False
    elif (button_id == 'btn2'):
        return True

#this will be triggered only when time intervals are enabled(when record is pressed)
@app.callback(
    [Output('graph-container', 'style'), Output('live-update-graph', 'figure')],
    [Input('interval-component', 'n_intervals')],
    prevent_initial_call=True
)
def live_speech_signal(n):
    global block
    fs = 44100
    block_size = 0.2  # window size in seconds
    mid_buf_size = int(fs * block_size)
    fig = go.Figure()
    x = np.arange(0, 2 * mid_buf_size, 2)
    data = np.frombuffer(block, np.int16)
    fig.add_trace(go.Scatter(x=x, y=data))

    fig.update_layout(
        autosize=False,
        width=1500,
        height=600,
        yaxis=dict(
            titlefont=dict(size=30),
            range=(-10000, 10000),
        ),
        xaxis=dict(
            range=(0, mid_buf_size),
        )
    )
    return {'display': 'block'}, fig





# Page 3 callbacks
@app.callback(
    [Output('display-text', 'style'), Output('graphs', 'style'), Output('graph1', 'figure'),
     Output('graph2', 'figure')],
    [Input('url', 'pathname')],
)
def display_page(pathname):
    if pathname == '/score':
        fig, fig1 = make_figures()
        return {'display': 'none'}, {'display': 'block'},fig,fig1



if __name__ == "__main__":
    app.run_server(debug=True)
