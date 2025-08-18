# 3D Camera-TV Calibration Visualization

This repository provides comprehensive 3D visualization tools for camera-TV calibration systems, offering both traditional matplotlib and modern interactive Plotly implementations.

## 🎯 Overview

The visualization system displays:
- **Multiple camera types**: Fisheye OMS, Pinhole RealSense, CCTV, TFT-Helios depth cameras
- **TV screens**: Multiple displays with grid patterns and coordinate systems
- **Spatial relationships**: Camera poses, TV positions, and coordinate transformations
- **Interactive exploration**: Zoom, rotate, pan, and inspect the 3D scene

## 🚀 Quick Start

### Option 1: Enhanced Interactive Visualization (Recommended)
```bash

# Run interactive Plotly visualization
python3 plot_data_collection_env_3D_plotly.py
```

### Option 2: Traditional Matplotlib Visualization
```bash
# Run original matplotlib visualization
python3 plot_data_collection_env_3D.py
```

## 📊 Visualization Comparison

| Feature | Matplotlib | Plotly |
|---------|------------|--------|
| **Interactivity** | Basic toolbar | Smooth 3D controls |
| **Visual Quality** | Standard | Professional WebGL |
| **Performance** | Slow for complex scenes | Fast hardware acceleration |
| **Hover Information** | None | Rich tooltips |
| **Export Options** | PNG, SVG | HTML + PNG/SVG/PDF |
| **Web Compatibility** | None | Full browser support |
| **Mobile Support** | No | Responsive design |
| **File Size** | Small | Larger (includes data) |

## 📁 File Structure

```
visualization_3d/
├── 📊 Matplotlib Implementation
│   ├── plot_data_collection_env_3D.py      # Main matplotlib script
│   └── utils/plot_3d.py                    # Matplotlib utilities
│
├── 🚀 Plotly Implementation  
│   ├── plot_data_collection_env_3D_plotly.py  # Main Plotly script
│   └── utils/plot_3d_plotly.py               # Plotly utilities
│
├── 📁 Data & Outputs
│   ├── camera_tv_params/                  # Calibration data
│   └── outputs/                           # Generated visualizations
│
└── 🔧 Utilities
    └── utils/transform.py                  # Coordinate transformations
```

## 🎮 Interactive Controls (Plotly)

| Action | Control |
|--------|---------|
| Rotate 3D view | Left click + drag |
| Pan view | Right click + drag |
| Zoom in/out | Scroll wheel |
| Reset view | Double click |
| Show/hide elements | Click legend items |
| Camera controls | Toolbar (top-right) |
| Save image | Right-click menu |

## 🎨 Visualization Features

### Camera Visualization
- **Color-coded cameras** by type (Fisheye, Pinhole, Depth)
- **3D camera frustums** showing field of view
- **Coordinate axes** for each camera
- **Hover information** with camera details

### TV Screen Visualization  
- **Semi-transparent screens** for better depth perception
- **Grid patterns** with numbered cells
- **Screen dimensions** and orientations
- **Coordinate systems** for each display

### Scene Features
- **Professional lighting** and materials
- **Configurable view angles** and zoom levels
- **Legend controls** for element visibility
- **Export capabilities** for presentations

## 🔧 Customization

### Adding New Cameras
```python
viz.add_camera(
    R=rotation_matrix,           # 3x3 rotation matrix
    tvec=translation_vector,     # 3x1 translation vector  
    cam_color='blue',           # Color name or hex
    label='My Camera',          # Legend label
    cam_size=100               # Size scaling factor
)
```

### Adding New Screens
```python
viz.add_tv_screen(
    R=rotation_matrix,
    tvec=translation_vector,
    monitor_mm=(width, height),     # Physical dimensions
    label='My Screen',
    screen_color='rgba(200,200,200,0.3)',  # RGBA color
    border_color='black'
)
```

## 📦 Dependencies

### Core Requirements
- `numpy` - Numerical computations
- `opencv-python` - Computer vision operations
- `matplotlib` - Traditional plotting (matplotlib version)

### Enhanced Requirements (Plotly)
- `plotly>=5.17.0` - Interactive 3D visualization
- `kaleido>=0.2.1` - Static image export
- `pandas>=1.5.0` - Data handling (optional)

## 🐛 Troubleshooting

### Common Issues
1. **Import errors**: Install missing dependencies with `pip3 install -r requirements_plotly.txt`
2. **Browser not opening**: Manually open `outputs/interactive_calibration_result.html`
3. **Performance issues**: Reduce camera/screen sizes for large datasets
4. **Export failures**: Ensure `kaleido` is properly installed for image export

### Platform-Specific Notes
- **Linux**: May require additional system packages for browser integration
- **Windows**: Should work out-of-the-box with most browsers
- **macOS**: Tested with Safari, Chrome, and Firefox
- **Headless environments**: Use static image export only

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Animation support for camera movements
- Additional camera/screen geometries  
- Custom color schemes and themes
- Integration with other 3D libraries
- Performance optimizations

## 📝 Usage Examples

### Basic Usage
```python
from utils.plot_3d_plotly import PlotlyVisualizer3D

# Create visualizer
viz = PlotlyVisualizer3D(title="My 3D Scene")

# Add elements
viz.add_camera(R, tvec, cam_color='red', label='Camera 1')
viz.add_tv_screen(R, tvec, (1920, 1080), label='Display 1')

# Show interactive visualization
viz.show(save_html=True)
```

### Advanced Customization
```python
# Custom scene setup
viz = PlotlyVisualizer3D(title="Custom Calibration")

# Modify layout
viz.fig.update_layout(
    scene=dict(
        camera=dict(eye=dict(x=2, y=2, z=2)),
        bgcolor='lightgray'
    )
)

# Export options
viz.save_image('my_scene.png', width=1920, height=1080)
```

## 📄 License

This project follows the same license as the parent repository.
