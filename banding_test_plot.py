import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import os
import base64
from dash.dependencies import Input, Output

# Create figure
os.chdir('H:/Ender/banding/20210512/nhs199_gryfn_1546/')
print(os.getcwd())
app = dash.Dash(__name__)
image_filename = 'H:/Ender/banding/20210512/nhs199_gryfn_1546/assets/raw_15215_rd_or_grifyn.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

app.layout = html.Div([
    html.H2(id='header',children='yoyoyo mother fucker'),
    html.Img(id='mother_fucker',src='data:image/png;base64,{}'.format(encoded_image.decode())),
    dcc.Dropdown(id="targets",
                options=[
                    {"label": "grifyn", "value": "grifyn"},
                    {"label": "Lab_5p", "value": "Lab_5p"},
                    {"label": "Lab_50p", "value": "Lab_50p"},
                    {"label": "Lab_80p", "value": "Lab_80p"},
                    {"label": "Type8_11", "value": "Type8_11p"},
                    {"label": "Type8_30p", "value": "Type8_30p"},
                    {"label": "Type8_56p", "value": "Type8_56p"}],
                multi=False,
                value='Lab_5p',
                style={'width': "40%"}
                ),
])

@app.callback(
    [Output(component_id='mother_fucker', component_property='src'),
    Output(component_id='header', component_property='children')],
    [Input(component_id='targets', component_property='value')]
)
def update(targets):
    h=targets
    image_filename = 'H:/Ender/banding/20210512/nhs199_gryfn_1546/assets/raw_15215_rd_or_'+targets+'.png'
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    src='data:image/png;base64,{}'.format(encoded_image.decode())
    return src,h


# app.layout=update_layout()
if __name__ == '__main__':

    app.run_server(debug=False)