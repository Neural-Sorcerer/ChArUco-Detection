# === Standard Libraries ===
import os
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

# === Third-Party Libraries ===
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Set default renderer for better compatibility
pio.renderers.default = "browser"

GRID_ROWS = 6
GRID_COLS = 8


class PlotlyVisualizer3D:
    """
    A class for creating interactive 3D visualizations using Plotly.
    Provides better interactivity and visual quality compared to matplotlib.
    """
    
    def __init__(self, title: str = "3D Calibration Visualization"):
        self.fig = go.Figure()
        self.title = title
        self.traces = []
        self._setup_layout()
    
    def _setup_layout(self):
        """Setup the 3D scene layout with proper axes and styling."""
        self.fig.update_layout(
            title={
                'text': self.title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            scene=dict(
                xaxis=dict(
                    title='X (mm)',
                    range=[-2100, 2100],
                    showgrid=True,
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor='white'
                ),
                yaxis=dict(
                    title='Y (mm)',
                    range=[-1800, 1500],
                    showgrid=True,
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor='white'
                ),
                zaxis=dict(
                    title='Z (mm)',
                    range=[-350, 2800],
                    showgrid=True,
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor='white'
                ),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            width=1400,
            height=1000,
            margin=dict(l=0, r=0, t=50, b=0),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
        )
    
    def add_camera(
        self,
        R: np.ndarray,
        tvec: np.ndarray,
        cam_color: str = 'blue',
        label: str = "Camera",
        cam_size: float = 100
    ):
        """Add a camera visualization to the scene."""
        # Camera frustum points in local coordinates
        r = cam_size
        r1_5 = r * 1.5
        r2_0 = r * 2
        r3_0 = r * 3
        
        # Define camera geometry
        local_points = np.array([
            [r, -r, -r2_0],    # x1
            [r, r, -r2_0],     # x2
            [-r, -r, -r2_0],   # x3
            [-r, r, -r2_0],    # x4
            [r, -r, r2_0],     # x5
            [r, r, r2_0],      # x6
            [-r, -r, r2_0],    # x7
            [-r, r, r2_0],     # x8
            [r1_5, -r1_5, r3_0],  # x9
            [r1_5, r1_5, r3_0],   # x10
            [-r1_5, -r1_5, r3_0], # x11
            [-r1_5, r1_5, r3_0]   # x12
        ]).T
        
        # Transform to world coordinates
        world_points = (R @ local_points) + tvec.reshape(3, 1)
        
        # Define camera wireframe connections
        connections = [
            [0, 1], [0, 2], [3, 2], [1, 3],  # back face
            [4, 5], [4, 6], [7, 6], [5, 7],  # front face
            [0, 4], [1, 5], [2, 6], [3, 7],  # connecting edges
            [8, 9], [8, 10], [11, 10], [9, 11],  # lens face
            [4, 8], [5, 9], [6, 10], [7, 11]   # lens connections
        ]
        
        # Create wireframe traces
        for i, (start, end) in enumerate(connections):
            self.fig.add_trace(go.Scatter3d(
                x=[world_points[0, start], world_points[0, end]],
                y=[world_points[1, start], world_points[1, end]],
                z=[world_points[2, start], world_points[2, end]],
                mode='lines',
                line=dict(color=cam_color, width=3),
                showlegend=(i == 0),  # Only show legend for first trace
                name=label,
                hoverinfo='skip'
            ))
        
        # Add camera center point
        center = tvec.flatten()
        self.fig.add_trace(go.Scatter3d(
            x=[center[0]],
            y=[center[1]],
            z=[center[2]],
            mode='markers',
            marker=dict(
                size=8,
                color=cam_color,
                symbol='circle'
            ),
            showlegend=False,
            hovertemplate=f'<b>{label}</b><br>X: %{{x:.1f}} mm<br>Y: %{{y:.1f}} mm<br>Z: %{{z:.1f}} mm<extra></extra>'
        ))
        
        # Add coordinate axes for camera
        self._add_coordinate_axes(R, tvec, length=180, linewidth=2, show_legend=False)
    
    def _add_coordinate_axes(
        self,
        R: np.ndarray,
        tvec: np.ndarray,
        length: float = 200,
        linewidth: float = 3,
        show_legend: bool = True
    ):
        """Add coordinate axes at the given pose."""
        origin = tvec.flatten()
        
        # X-axis (red)
        x_end = origin + (R @ np.array([length, 0, 0]))
        self.fig.add_trace(go.Scatter3d(
            x=[origin[0], x_end[0]],
            y=[origin[1], x_end[1]],
            z=[origin[2], x_end[2]],
            mode='lines',
            line=dict(color='red', width=linewidth),
            showlegend=show_legend,
            name='X-axis' if show_legend else None,
            hoverinfo='skip'
        ))
        
        # Y-axis (green)
        y_end = origin + (R @ np.array([0, length, 0]))
        self.fig.add_trace(go.Scatter3d(
            x=[origin[0], y_end[0]],
            y=[origin[1], y_end[1]],
            z=[origin[2], y_end[2]],
            mode='lines',
            line=dict(color='green', width=linewidth),
            showlegend=show_legend,
            name='Y-axis' if show_legend else None,
            hoverinfo='skip'
        ))
        
        # Z-axis (blue)
        z_end = origin + (R @ np.array([0, 0, length]))
        self.fig.add_trace(go.Scatter3d(
            x=[origin[0], z_end[0]],
            y=[origin[1], z_end[1]],
            z=[origin[2], z_end[2]],
            mode='lines',
            line=dict(color='blue', width=linewidth),
            showlegend=show_legend,
            name='Z-axis' if show_legend else None,
            hoverinfo='skip'
        ))
    
    def add_tv_screen(
        self,
        R: np.ndarray,
        tvec: np.ndarray,
        monitor_mm: Tuple[float, float],
        label: str = "TV Screen",
        screen_color: str = 'rgba(200, 200, 200, 0.3)',
        border_color: str = 'black'
    ):
        """Add a TV screen visualization to the scene."""
        width_mm, height_mm = monitor_mm
        
        # Define screen corners in local coordinates
        screen_local = np.array([
            [0, 0, 0],                    # bottom-left
            [width_mm, 0, 0],             # bottom-right
            [width_mm, height_mm, 0],     # top-right
            [0, height_mm, 0],            # top-left
        ]).T
        
        # Transform to world coordinates
        screen_world = (R @ screen_local) + tvec.reshape(3, 1)
        
        # Add screen surface
        self.fig.add_trace(go.Mesh3d(
            x=screen_world[0, :],
            y=screen_world[1, :],
            z=screen_world[2, :],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color=screen_color,
            opacity=0.3,
            showlegend=True,
            name=label,
            hovertemplate=f'<b>{label}</b><br>Size: {width_mm:.0f} x {height_mm:.0f} mm<extra></extra>'
        ))
        
        # Add screen border
        border_points = np.column_stack([screen_world, screen_world[:, 0:1]])  # Close the loop
        self.fig.add_trace(go.Scatter3d(
            x=border_points[0, :],
            y=border_points[1, :],
            z=border_points[2, :],
            mode='lines',
            line=dict(color=border_color, width=4),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add grid lines and labels
        self._add_screen_grid(R, tvec, monitor_mm)
        
        # Add coordinate axes for TV
        self._add_coordinate_axes(R, tvec, length=300, linewidth=3, show_legend=False)
    
    def _add_screen_grid(self, R: np.ndarray, tvec: np.ndarray, monitor_mm: Tuple[float, float]):
        """Add grid lines and cell labels to the TV screen."""
        width_mm, height_mm = monitor_mm
        
        # Vertical grid lines
        for i in range(1, GRID_COLS):
            x = i * (width_mm / GRID_COLS)
            pt1 = np.array([[x], [0], [0]])
            pt2 = np.array([[x], [height_mm], [0]])
            p1w = (R @ pt1) + tvec.reshape(3, 1)
            p2w = (R @ pt2) + tvec.reshape(3, 1)
            
            self.fig.add_trace(go.Scatter3d(
                x=[p1w[0, 0], p2w[0, 0]],
                y=[p1w[1, 0], p2w[1, 0]],
                z=[p1w[2, 0], p2w[2, 0]],
                mode='lines',
                line=dict(color='#246B6D', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Horizontal grid lines
        for j in range(1, GRID_ROWS):
            y = j * (height_mm / GRID_ROWS)
            pt1 = np.array([[0], [y], [0]])
            pt2 = np.array([[width_mm], [y], [0]])
            p1w = (R @ pt1) + tvec.reshape(3, 1)
            p2w = (R @ pt2) + tvec.reshape(3, 1)
            
            self.fig.add_trace(go.Scatter3d(
                x=[p1w[0, 0], p2w[0, 0]],
                y=[p1w[1, 0], p2w[1, 0]],
                z=[p1w[2, 0], p2w[2, 0]],
                mode='lines',
                line=dict(color='#246B6D', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add cell labels
        cell_positions_x = []
        cell_positions_y = []
        cell_positions_z = []
        cell_labels = []
        
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                cx = (col + 0.5) * (width_mm / GRID_COLS)
                cy = (row + 0.5) * (height_mm / GRID_ROWS)
                pt = np.array([[cx], [cy], [0]])
                ptw = (R @ pt) + tvec.reshape(3, 1)
                
                cell_positions_x.append(ptw[0, 0])
                cell_positions_y.append(ptw[1, 0])
                cell_positions_z.append(ptw[2, 0])
                cell_labels.append(str(GRID_COLS * row + col + 1))
        
        # Add text annotations for grid cells
        self.fig.add_trace(go.Scatter3d(
            x=cell_positions_x,
            y=cell_positions_y,
            z=cell_positions_z,
            mode='text',
            text=cell_labels,
            textfont=dict(size=8, color='black'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    def show(self, save_html: bool = False, filename: str = "3d_visualization.html"):
        """Display the interactive 3D visualization."""
        if save_html:
            os.makedirs("outputs", exist_ok=True)
            self.fig.write_html(f"outputs/{filename}")
            print(f"Saved interactive visualization to outputs/{filename}")
        
        self.fig.show()
    
    def save_image(self, filename: str = "3d_visualization.png", width: int = 1400, height: int = 1000):
        """Save the visualization as a static image."""
        os.makedirs("outputs", exist_ok=True)
        self.fig.write_image(f"outputs/{filename}", width=width, height=height)
        print(f"Saved static image to outputs/{filename}")
