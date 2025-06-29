import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Replace with your actual data
IPTG_concentration = [5.8,  15.6, 23.0, 26.3, 35.0, 52.5, 70.0, 93.3, 140.0, 210.0]

IPTG_skew_3000 = [0.2705,  0.1951, -0.0495, 0.0762, -0.9452, -0.2613,
                  -0.6564, -0.6351, -0.4904, -0.4316]

IPTG_skew_5000 = [0.0789, 0.2000, -0.0297, 0.0889, -0.8578, -1.0050,
                  -0.4692, -0.6763, -0.3746, -0.4547]


TMG_concentration = [2.3, 9.3, 18.5, 31.3, 55.6, 62.5, 111.0]

TMG_skew_3000 = [1.828901104, 1.25619138, 0.6201, 0.3548, 0.558900541, 0.4468, 0.2963]

TMG_skew_5000 = [3.863348127, 1.1779, 0.5224, 0.344372203, 0.566315798, 0.4399, 0.2689]


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
    line=dict(width=2, dash='dash'),
    name=r"$IPTG - Exposure\ 3000$",
    legendgroup="IPTG"
))

fig.add_trace(go.Scatter(
    x=IPTG_concentration,
    y=IPTG_skew_5000,
    mode='markers+lines',
    marker=dict(symbol=markers[1], size=10, color=colors[0]),
    name=r"$IPTG - Exposure\ 5000$",
    legendgroup="iptg"
))

# TMG
fig.add_trace(go.Scatter(
    x=TMG_concentration,
    y=TMG_skew_3000,
    mode='markers+lines',
    line=dict(width=2, dash='dash'),
    marker=dict(symbol=markers[0], size=10, color=colors[1]),
    name=r"$TMG - Exposure\ 3000$",
    legendgroup="TMG"
))

fig.add_trace(go.Scatter(
    x=TMG_concentration,
    y=TMG_skew_5000,
    mode='markers+lines',
    marker=dict(symbol=markers[1], size=10, color=colors[1]),
    name=r"$TMG - Exposure\ 5000$",
    legendgroup="tmg"
))

fig.update_layout(
    title=r"$Skewness\ vs.\ Inducer\ Concentration$",
    xaxis=dict(
        title=r"$Inducer\ Concentration\ [\mu M]$",
        title_font=dict(size=24),
        showgrid=True,
        gridcolor='lightgray'
    ),
    yaxis=dict(
        title=r"$Skewness$",
        title_font=dict(size=24),
        showgrid=True,
        gridcolor='lightgray'
    ),
    legend=dict(
        x=0.7,
        y=0.9,
        font=dict(size=14)
    ),
    width=1000,
    height=600,
    template='plotly_white'
)

fig.write_html("./figures/skewness_vs_concentration.html", include_mathjax='cdn')
fig.show()