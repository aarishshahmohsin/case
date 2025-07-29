import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from plotly.subplots import make_subplots

TOLERANCE = 1e-06

# -------------------------
# Helper Functions
# -------------------------
def safe_get(df_row, variable_name, default=0.0):
    """Safely get variable value from dataframe row"""
    if variable_name in df_row.columns:
        val = df_row[variable_name].iloc[0] if len(df_row) > 0 else default
        return float(val)
    return float(default)

def value_to_color(value, color0, color1):
    """Linearly interpolate between two colors based on value (0-1)"""
    c0 = np.array(mcolors.to_rgb(color0))
    c1 = np.array(mcolors.to_rgb(color1))
    color = c0 + (c1 - c0) * value
    return f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'

def compute_reach(P, w, c, epsilon_P):
    """Calculate how many P points are correctly classified with margin"""
    # Calculate τ(s) = s^T w + c
    tau = P @ w + c
    # Count points where τ(s) >= epsilon_P
    consistent = tau >= epsilon_P
    return np.sum(consistent)

# -------------------------
# Load Data
# -------------------------
# df = pd.read_csv('./feaspump_variables.csv')
df = pd.read_csv('/home/aarish/case/intshift_vars_intshifting_20250729_091349.csv')
P = np.load('./P_Binary.npy')  # Shape (n, 3)
N = np.load('./N_Binary.npy')  # Shape (m, 3)

# Get epsilon values (if available)

from src.constants import epsilon_N, epsilon_P, epsilon_R


# -------------------------
# Prepare Animation Frames
# -------------------------
iterations = df['Iteration'].unique()
frames_sequence = []
for iteration in sorted(iterations):
    stages = df[df['Iteration'] == iteration]['Stage'].unique()
    for stage in stages:
        frames_sequence.append((iteration, stage))
    # if 'LP' in stages:
    #     frames_sequence.append((iteration, 'LP'))
    # if 'ROUNDED' in stages:
    #     frames_sequence.append((iteration, 'ROUNDED'))

# Precompute hyperplane grid
x_range = np.linspace(P[:, 0].min() - 1, P[:, 0].max() + 1, 20)
y_range = np.linspace(P[:, 1].min() - 1, P[:, 1].max() + 1, 20)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# Initialize hyperplane coefficients
last_w0, last_w1, last_w2, last_c = 0.0, 0.0, 1.0, 0.0

# Create figure
fig = go.Figure()

# -------------------------
# Create Animation Frames
# -------------------------
frames = []
for i, (iteration, stage) in enumerate(frames_sequence):
    row = df[(df['Iteration'] == iteration) & (df['Stage'] == stage)]
    
    # Update hyperplane parameters if available
    if stage == 'INITIAL' or stage == 'PRE_LP' and not row.empty:
        last_w0 = safe_get(row, 't_w_0', last_w0)
        last_w1 = safe_get(row, 't_w_1', last_w1)
        last_w2 = safe_get(row, 't_w_2', last_w2)
        last_c = safe_get(row, 't_c', last_c)

 
    
    w0, w1, w2, c = last_w0, last_w1, last_w2, last_c
    
    # Compute current reach
    total_P = len(P)
    reach = 0 
    for idx in range(len(P)):
        var_name = f't_x_{idx}'
        val = safe_get(row, var_name, 0)
        if val > (1-TOLERANCE):  # Essentially 1
            reach += 1
        
    reach_text = f"Reach: {reach}/{total_P}"
    
    # Create hyperplane and margin surfaces
    hyperplane_traces = []
    # if abs(w2) > 1e-6:
    # Compute the norm of the weight vector for proper margin distances
    w_norm = np.sqrt(w0**2 + w1**2 + w2**2)
    
    # Main hyperplane
    Z_main = -(w0 * X_grid + w1 * Y_grid - c) / (w2 + epsilon_R)
    hyperplane_traces.append(go.Surface(
        x=X_grid, y=Y_grid, z=Z_main,
        colorscale=[[0, 'gray'], [1, 'gray']],
        opacity=0.6,
        showscale=False,
        name='Hyperplane'
    ))
    
    # Positive margin (for P points)
    Z_pos = -(w0 * X_grid + w1 * Y_grid - epsilon_P -  c) / (w2 + epsilon_R)
    hyperplane_traces.append(go.Surface(
        x=X_grid, y=Y_grid, z=Z_pos,
        colorscale=[[0, 'red'], [1, 'red']],
        opacity=0.2,  # Low opacity as requested
        showscale=False,
        name='P Margin'
    ))
    
    # Negative margin (for N points)
    Z_neg = -(w0 * X_grid + w1 * Y_grid - c  + epsilon_N ) / (w2 + epsilon_R)
    hyperplane_traces.append(go.Surface(
        x=X_grid, y=Y_grid, z=Z_neg,
        colorscale=[[0, 'green'], [1, 'green']],
        opacity=0.2,  # Low opacity as requested
        showscale=False,
        name='N Margin'
    ))

    # Prepare point data with labels
    P_text = []
    P_colors = []
    for idx in range(len(P)):
        var_name = f't_x_{idx}'
        val = safe_get(row, var_name, 0)
        if val < TOLERANCE:  # Essentially 0
            color = 'black'
        elif val > (1-TOLERANCE):  # Essentially 1
            color = 'red'
        else:  # Fractional
            color = 'grey' 
        P_colors.append(color)
        P_text.append(f"x_{idx}: {val:.6f}")  # Label with variable index and value
    
    N_text = []
    N_colors = []
    for idx in range(len(N)):
        var_name = f't_y_{idx}'
        val = safe_get(row, var_name, 0)
        if val < TOLERANCE:  # Essentially 0
            color = 'green'
        elif val > (1-TOLERANCE):  # Essentially 1
            color = 'blue'
        else:  # Fractional
            color = 'yellow'
        N_colors.append(color)
        N_text.append(f"y_{idx}: {val:.6f}")  # Label with variable index and value
    
    # Create point traces with labels
    point_traces = [
        go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2],
            mode='markers',
            marker=dict(size=5, color=P_colors, opacity=0.8),
            name='P Points',
            text=P_text,  # Add labels
            hoverinfo='text+x+y+z'
        ),
        go.Scatter3d(
            x=N[:, 0], y=N[:, 1], z=N[:, 2],
            mode='markers',
            marker=dict(size=5, color=N_colors, opacity=0.8),
            name='N Points',
            text=N_text,  # Add labels
            hoverinfo='text+x+y+z'
        )
    ]
    
    # Combine all traces for this frame
    frame_data = point_traces + hyperplane_traces
    
    # Create frame with reach in title
    frame = go.Frame(
        data=frame_data,
        name=str(i),
        layout=go.Layout(
            title_text=f"Iteration {iteration} ({stage})<br>{reach_text}"
        )
    )
    frames.append(frame)

# Add initial traces
fig.add_trace(go.Scatter3d(
    x=P[:, 0], y=P[:, 1], z=P[:, 2],
    mode='markers',
    marker=dict(size=5, color='red', opacity=0.8),
    name='P Points',
    text=[f"x_{i}: 0.00" for i in range(len(P))],  # Initial labels
    hoverinfo='text+x+y+z'
))

fig.add_trace(go.Scatter3d(
    x=N[:, 0], y=N[:, 1], z=N[:, 2],
    mode='markers',
    marker=dict(size=5, color='blue', opacity=0.8),
    name='N Points',
    text=[f"y_{i}: 0.00" for i in range(len(N))],  # Initial labels
    hoverinfo='text+x+y+z'
))

# Add initial dummy hyperplane
fig.add_trace(go.Surface(
    x=[[0, 1], [0, 1]], y=[[0, 0], [1, 1]], z=[[0, 0], [0, 0]],
    opacity=0,
    showscale=False
))

# fig.add_trace(go.Surface(
#     x=[[0, 1], [0, 1]], y=[[0, 0], [1, 1]], z=[[0, 1], [0, 0]],
#     opacity=0,
#     showscale=False
# ))

# fig.add_trace(go.Surface(
#     x=[[0, 1], [0, 1]], y=[[0, 0], [1, 1]], z=[[1, 0], [0, 0]],
#     opacity=0,
#     showscale=False
# ))

# Add frames to figure
fig.frames = frames

# -------------------------
# Animation Controls
# -------------------------
animation_settings = {
    'frame': {'duration': 1000, 'redraw': True},
    'fromcurrent': True,
    'mode': 'immediate'
}

# Create slider
sliders = [{
    'active': 0,
    'steps': [
        {
            'args': [[f.name], animation_settings],
            'label': f'Iter: {frames_sequence[i][0]} ({frames_sequence[i][1]})',
            'method': 'animate'
        }
        for i, f in enumerate(fig.frames)
    ],
    'transition': {'duration': 300},
    'x': 0.1, 'y': 0,
    'len': 0.9,
    'currentvalue': {
        'prefix': 'Frame: ',
        'visible': True,
        'xanchor': 'right'
    }
}]

# Create play/pause buttons
buttons = [{
    'type': 'buttons',
    'buttons': [
        {
            'args': [None, animation_settings],
            'label': '▶ Play',
            'method': 'animate'
        },
        {
            'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}],
            'label': '⏸ Pause',
            'method': 'animate'
        },
        {
            'args': [[0], animation_settings],
            'label': '↺ Reset',
            'method': 'animate'
        }
    ],
    'x': 0.25, 'y': 1.15,
    'xanchor': 'right',
    'yanchor': 'top'
}]

# -------------------------
# Final Layout Configuration
# -------------------------
fig.update_layout(
    title='Feasibility Pump 3D Animation with Point Values',
    scene=dict(
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        zaxis_title='Feature 3',
        xaxis=dict(range=[P[:, 0].min()-1, P[:, 0].max()+1]),
        yaxis=dict(range=[P[:, 1].min()-1, P[:, 1].max()+1]),
        zaxis=dict(range=[P[:, 2].min()-1, P[:, 2].max()+1]),
        aspectmode='cube',
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    updatemenus=buttons,
    sliders=sliders,
    height=800,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Save as interactive HTML
fig.write_html("intshifting.html")
print("Interactive Plotly animation with point labels saved as feaspump_3d_with_labels.html")