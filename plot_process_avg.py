import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from collections import defaultdict
from consts import BASE_FOLDER, MASKS_FOLDER, PICTURE_FOLDER
from fit_functions import model, hillEq

def plot_multiple_res_dicts_grouped(res_dict_array, labels=None, save_path=None, 
                                   title="Hill Coefficient Fits - Multiple Datasets"):
    """
    Plots multiple res_dict objects with different markers and colors using Plotly.
    Each res_dict gets one color, with different markers for different exposures within that dataset.
    Labels in the format: "{label} - exposure {exposure_key}"
    
    Parameters:
    -----------
    res_dict_array : list of dict
        Array of res_dict objects
    labels : list of str, optional
        Labels for each dataset. If None, uses "Dataset 1", "Dataset 2", etc.
    save_path : str, optional
        Path to save the figure (HTML format)
    title : str
        Title for the plot
    """
    
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(res_dict_array))]
    
    # Define different marker symbols for different exposures within each dataset
    markers = ['circle', 'circle-open', 'star', 'square', 'triangle-up', 'triangle-down', 'triangle-left',
               'triangle-right', 'diamond', 'pentagon', 'hexagon']
    
    # Get colors using Plotly's color scale
    colors = px.colors.qualitative.Set1
    if len(res_dict_array) > len(colors):
        # Extend colors if needed
        colors = colors * (len(res_dict_array) // len(colors) + 1)
    
    # Create figure
    fig = go.Figure()
    
    # Process each dataset
    for dataset_idx, res_dict in enumerate(res_dict_array):
        label = labels[dataset_idx]
        if label=="IPTG":
            UpperBound=2.5
        else:
            UpperBound=7.5

        # Each dataset gets one color
        dataset_color = colors[dataset_idx % len(colors)]
        
        # Get all exposures for this dataset to assign markers
        exposures = sorted(res_dict.keys())
        
        for exp_idx, exp in enumerate(exposures):
            try:
                marker_symbol = markers[exp_idx % len(markers)]  # Different marker for each exposure
                print(exp)
                x_exp = res_dict[exp]['x']
                y_exp = res_dict[exp]['y']
                error_exp = res_dict[exp]['error']
                if label=="IPTG":
                    y_exp = y_exp[(x_exp <= 50) | (x_exp >= 60)]
                    x_exp = x_exp[(x_exp <= 50) | (x_exp >= 60)]
                
                # Calculate R (normalized response)
                # R = (y_exp - y_exp[x_exp.argmin()]) / (y_exp[y_exp.argmax()] - y_exp[x_exp.argmin()])

                initial_guess =  [7, 100, 0.5, 2, 0, 0.05]  # D, R, C, n, Y0
                lower_bounds = [0, 0, 0, 1, -1000, -1000]  # R >= 0 enforced here
                upper_bounds = [np.inf] * 3 + [UpperBound, np.inf, np.min(x_exp)]
                # initial_guess_hill = [1, 6]  # Kd,n
                # lower_bounds_hill = [0, 0.9]  # R >= 0 enforced here
                # upper_bounds_hill = [np.inf, 9]

                # params_hill, cov_hill = curve_fit(
                #     hillEq, x_exp, R,
                #     p0=initial_guess_hill,
                #     bounds=(lower_bounds_hill, upper_bounds_hill),
                #     maxfev=100000
                # )

                params, cov = curve_fit(
                    model, x_exp, y_exp,
                    p0=initial_guess,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=10000
                )
                print("plotting...")

                # Kd_hillfit, n_hillfit = params_hill
                D_fit, R_fit, C_fit, n_fit, Y0_fit, X0_fit = params

                # Create label in the requested format
                legend_label = fr"${label} - exposure \ {exp}$"
                params_legend_label = (f'Exp {exp} | D={D_fit:.1f}, '
                                      f'R={R_fit:.1f}'
                                      f'C={C_fit:.4f}'
                                      f'n={n_fit:.1f}'
                                      f'Y₀={Y0_fit:.2f}')
                print(f"{label} - {exp}: ", params_legend_label)
                # Add scatter points
                fig.add_trace(go.Scatter(
                    x=x_exp, 
                    y=y_exp,
                    mode='markers',
                    marker=dict(
                        symbol=marker_symbol,
                        size=10,
                        color=dataset_color,
                        # line=dict(width=1, color='black')
                    ),
                    # error_y=dict(
                    #     type='constant',
                    #     value=0.1
                    # ),
                    error_y=dict(
                        type='data',  # value of error bar given in data coordinates
                        array=error_exp,
                        visible=True),
                    name=legend_label,
                    legendgroup=f"dataset_{dataset_idx}",  # Group by dataset
                    showlegend=True
                ))
                
                # Add fitted curve
                # x_hill = np.linspace(min(x_exp), max(x_exp), 500)
                # R_hill = hillEq(x_hill, *params_hill)

                x_fit = np.linspace(min(x_exp), max(x_exp), 500)
                y_fit = model(x_fit, *params)


                fig.add_trace(go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode='lines',
                    line=dict(color=dataset_color, width=2, dash='dash' if exp != 3000 else 'solid'),
                    name=f"Fit: {legend_label}",
                    legendgroup=f"dataset_{dataset_idx}",
                    showlegend=False,  # Don't show fit lines in legend to avoid clutter
                    # hovertemplate=f"Fit: {legend_label}<br>Kd: {Kd_hillfit:.2f}<br>n: {n_hillfit:.2f}<extra></extra>"
                ))
                
            except Exception as e:
                print(f"Error processing {label}, exposure {exp}: {e}")
                continue
    
    # Update layout
    fig.update_layout(
        # title=dict(
        #     text=title,
        #     x=0.5,
        #     font=dict(size=16)
        # ),
        xaxis=dict(
            title=r"$Inducer \ Concentration \ [\mu M]$",
            title_font=dict(size=28),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title=r"$Intensity \ [\#] $",
            title_font=dict(size=28),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        legend=dict(
            x=0.65,
            y=0.05,
            font=dict(size=15)
        ),
        width=1200,
        height=650,
        template='plotly_white'
    )
    
    # Show the plot
    # fig.write_image(save_path)
    fig.write_html(save_path, include_mathjax='cdn')
    fig.show()
    
    return fig
