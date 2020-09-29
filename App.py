import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output,State
import dash_bootstrap_components as dbc


import plotly.graph_objects as go

import sounddevice as sd
from scipy.io.wavfile import write

import pandas as pd 
import numpy as np
from subprocess import call
import json

def App(recall_score,precision_score,Df1,Df2):

	X_Rlist=Df1['x'].to_numpy()
	Y_Rlist=Df1['y'].to_numpy()
	X_Plist=Df2['x'].to_numpy()
	Y_Plist=Df2['y'].to_numpy()
	fig = go.Figure()
	fig1 = go.Figure()

	fig.add_trace(go.Bar(
	    x=["Precision Score", "Recall Score"],
	    y=[precision_score,recall_score]
	))

	fig1.add_trace(go.Scatter(x=X_Rlist, y=Y_Rlist,
                    mode='lines+markers',
                    name='recall score'))

	fig1.add_trace(go.Scatter(x=X_Plist, y=Y_Plist,
                    mode='lines+markers',
                    name='precision score'))


	fig.update_layout(
	    autosize=False,
	    width=1500,
	    height=600,
	    yaxis=dict(
	        title_text="Percentage score",
	        tickmode="array",
	        titlefont=dict(size=30),
	        range=(0,100),
	    )
	)

	fig1.update_layout(
	    autosize=False,
	    width=1500,
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

	fig.update_yaxes(automargin=True)

	return fig,fig1
text_markdown = "\t"
with open('ena_oneiro.txt') as this_file:
    for a in this_file.read():
        if "\n" in a:
            text_markdown += "\n \t"
        else:
            text_markdown += a	

app = dash.Dash(__name__,suppress_callback_exceptions=True,
	external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css',
		{
	        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
	        'rel': 'stylesheet',
	        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
	        'crossorigin': 'anonymous'
    	}
    ]
)


url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

layout_index= html.Div([
		dbc.Row(dbc.Col(html.H1("Select a text for reading"))),
		dbc.Row(
            [
				dbc.Col(html.A(html.Button('Μια φορά',id='text1',n_clicks=0),href='/page-1')),
				dbc.Col(html.A(html.Button('Ένα όνειρο',id='text2',n_clicks=0),href='/page-2')),
				dbc.Col(html.A(html.Button('Γελαστή οικογένεια',id='text3',n_clicks=0),href='/page-3')),
				dbc.Col(html.A(html.Button('Καλοκαιρινή εργασία',id='text4',n_clicks=0),href='/page-4')),
				dbc.Col(html.A(html.Button('Είμαι καλά',id='text5',n_clicks=0),href='/page-5'))
			]
		),
])
	


layout_page_1 = html.Div([
		dcc.Markdown(id='text'),
		html.Button('Record', id='btn', n_clicks=0),
		html.Button('Stop',id='btn2',n_clicks=0),
		html.Div(id='output',children='Hit the button to update'),
		dcc.Link('Show score',href='/score')
	
])

layout_page_2 = html.Div([
	dcc.Graph(
	    	id='graph1'
	),
	dcc.Graph(
	    	id='graph2'
    )
])

#index layout
app.layout = url_bar_and_content_div

#"complete" layout
app.validation_layout = html.Div([
    url_bar_and_content_div,
    layout_index,
    layout_page_1,
    layout_page_2,
])

#Index callbacks
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == "/page-1":
        return layout_page_1
    elif pathname == "/page-2":
    	return layout_page_1
    elif pathname == "/page-3":
    	return layout_page_1	
    elif pathname == "/page-4":
    	return layout_page_1
    elif pathname == "/page-5":
    	return layout_page_1
    elif pathname == "/score":
    	return layout_page_2
    else:
        return layout_index

#Page 1 callbacks
@app.callback(
	Output('text','children'),
	[Input('url','pathname')]
)
def text(pathname):
	text_markdown="\t"
	if pathname == "/page-1":
		with open('mia_fora.txt') as this_file:
			for a in this_file.read():
				if "\n" in a:
					text_markdown += "\n \t"
				else:
					text_markdown += a	
		with open('config.json', 'r') as file:
			json_data = json.load(file)
			json_data['reference_text'] = "mia_fora.txt"
		with open('config.json', 'w') as file:
			json.dump(json_data, file)
	elif pathname == "/page-2":
		with open('ena_oneiro.txt') as this_file:
			for a in this_file.read():
				if "\n" in a:
					text_markdown += "\n \t"
				else:
					text_markdown += a	
		with open('config.json', 'r') as file:
			json_data = json.load(file)
			json_data['reference_text'] = "ena_oneiro.txt"
		with open('config.json', 'w') as file:
			json.dump(json_data, file)
	elif pathname == "/page-3":
		with open('gelasti_oikogeneia.txt') as this_file:
			for a in this_file.read():
				if "\n" in a:
					text_markdown += "\n \t"
				else:
					text_markdown += a	
		with open('config.json', 'r') as file:
			json_data = json.load(file)
			json_data['reference_text'] = "gelasti_oikogeneia.txt"
		with open('config.json', 'w') as file:
			json.dump(json_data, file)
	elif pathname == "/page-4":
		with open('kalokairini_ergasia.txt') as this_file:
			for a in this_file.read():
				if "\n" in a:
					text_markdown += "\n \t"
				else:
					text_markdown += a	
		with open('config.json', 'r') as file:
			json_data = json.load(file)
			json_data['reference_text'] = "kalokairini_ergasia.txt"
		with open('config.json', 'w') as file:
			json.dump(json_data, file)
	elif pathname == "/page-5":
		with open('eimai_kala.txt') as this_file:
			for a in this_file.read():
				if "\n" in a:
					text_markdown += "\n \t"
				else:
					text_markdown += a	
		with open('config.json', 'r') as file:
			json_data = json.load(file)
			json_data['reference_text'] = "eimai_kala.txt"
		with open('config.json', 'w') as file:
			json.dump(json_data, file)
	return text_markdown 

@app.callback(
	Output('output', 'children'),
	[Input('btn','n_clicks')],
	prevent_initial_call=True

)
def update(n):
	fs = 44100  # Sample rate
	seconds = 20 # Duration of recording
	myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype=np.int16)
	sd.wait()  # Wait until recording is finished
	write('output.wav', fs, myrecording)  # Save as WAV file
	return 'Finished'



#Page 2 callbacks
@app.callback(
	[Output('graph1','figure'),Output('graph2','figure')],
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/score':
    	script_path = '/home/sofia/pythonenv2/lib/python3.6/site-packages/main.py'
    	call(["python3", script_path])
    	df = pd.read_pickle("score.pkl")
    	rec = df.loc[0,'recall']
    	pre = df.loc[0,'precision']
    	df1 = pd.read_pickle("recall_temporal.pkl")
    	df2 = pd.read_pickle("precision_temporal.pkl")
    	fig,fig1=App(rec,pre,df1,df2)
    	return fig,fig1
   


if __name__ == "__main__":
    app.run_server(debug=True)