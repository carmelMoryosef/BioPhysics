import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Replace with your actual data
IPTG_concentration = [5.8,  15.6, 23.0, 26.3, 35.0, 52.5, 70.0, 93.3, 140.0, 210.0]
IPTG_5000_error = [0.142616484966217,0.157788302303604,0.122782615302292,0.183089403866874,0.153697105249344,0.166285846741594,0.114457693145535,0.142616484966217,0.166285846741594,0.144339350596668]
IPTG_3000_error = [0.142858835707979,0.158779724662815,0.122782615302292,0.183603048164397,0.114582854399718,0.072389106332455,0.156176417701713,0.103602713097246,0.168235516246275,0.145608915360224]
IPTG_skew_3000 = [0.2705,  0.1951, -0.0495, 0.0762, -0.9452, -0.2613,
                  -0.6564, -0.6351, -0.4904, -0.4316]

IPTG_skew_5000 = [0.0789, 0.2000, -0.0297, 0.0889, -0.8578, -0.41,
                  -0.4692, -0.6763, -0.3746, -0.4547]


TMG_concentration = [2.3, 9.3, 18.5, 31.3, 55.6, 62.5, 111.0]

TMG_skew_3000 = [1.828901104, 1.25619138, 0.6201, 0.3548, 0.558900541, 0.4468, 0.2963]
TMG_3000_error = [0.12582272895995,0.128037854305006,0.16984561651928,0.117988414843195,0.129100460680779,0.14990849688398,0.123248008618293]

TMG_skew_5000 = [3.863348127, 1.1779, 0.5224, 0.344372203, 0.566315798, 0.4399, 0.2689]
TMG_5000_error = [0.193657074424043,0.14664904716909,0.167840095810805,0.15250092485016,0.130374251458174,0.150757893906988,0.123092289991961]

# Settings
markers = ['circle', 'circle-open', 'star', 'square', 'triangle-up', 'triangle-down', 'triangle-left',
               'triangle-right', 'diamond', 'pentagon', 'hexagon']
colors = px.colors.qualitative.Set1

fig = go.Figure()

# IPTG
fig.add_trace(go.Scatter(
    x=IPTG_concentration,
    y=IPTG_skew_3000,
    mode='markers+lines',
    marker=dict(symbol=markers[0], size=10, color=colors[0]),

    name=r"$IPTG - Exposure\ 3000$",
    legendgroup="IPTG",
    error_y=dict(
            type='data',  # value of error bar given in data coordinates
            array=IPTG_3000_error,
            visible=True)
))

fig.add_trace(go.Scatter(
    x=IPTG_concentration,
    y=IPTG_skew_5000,
    mode='markers+lines',
    marker=dict(symbol=markers[1], size=10, color=colors[0]),
    line=dict(width=2, dash='dash'),
    name=r"$IPTG - Exposure\ 5000$",
    legendgroup="iptg",
    error_y=dict(
            type='data',  # value of error bar given in data coordinates
            array=IPTG_5000_error,
            visible=True)
))

# TMG
fig.add_trace(go.Scatter(
    x=TMG_concentration,
    y=TMG_skew_3000,
    mode='markers+lines',

    marker=dict(symbol=markers[0], size=10, color=colors[1]),
    name=r"$TMG - Exposure\ 3000$",
    legendgroup="TMG",
    error_y=dict(
            type='data',  # value of error bar given in data coordinates
            array=TMG_3000_error,
            visible=True)
))

fig.add_trace(go.Scatter(
    x=TMG_concentration,
    y=TMG_skew_5000,
    mode='markers+lines',
    marker=dict(symbol=markers[1], size=10, color=colors[1]),
    line=dict(width=2, dash='dash'),
    name=r"$TMG - Exposure\ 5000$",
    legendgroup="tmg",
    error_y=dict(
            type='data',  # value of error bar given in data coordinates
            array=TMG_5000_error,
            visible=True)
    ))

fig.update_layout(
    title=r"$Skewness\ vs.\ Inducer\ Concentration$",
    xaxis=dict(
        title=r"$Inducer\ Concentration\ [\mu M]$",
        title_font=dict(size=34),
        showgrid=True,
        gridcolor='lightgray',
        tickfont = dict(size=17)
    ),
    yaxis=dict(
        title=r"$Skewness$",
        title_font=dict(size=34),
        showgrid=True,
        gridcolor='lightgray',
        tickfont = dict(size=17)
    ),
    legend=dict(
        x=0.7,
        y=0.9,
        font=dict(size=16)
    ),
    width=1000,
    height=600,
    template='plotly_white'
)

fig.write_html("./figures/skewness_vs_concentration.html", include_mathjax='cdn')
fig.show()