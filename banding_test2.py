import plotly.graph_objects as go
import numpy as np
from pandas import read_csv as read_csv

df=read_csv('H:/Ender/banding/20210605/SVC1024i_20210605/results/PREDiCT_good/Gryfn 11p 2021_average.txt',sep='\t',names=['wavelength','targets'])
# print(df.keys())
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['wavelength'], y=df['targets'],
                    mode='lines+markers',
                    name='lines+markers'))
fig.show()