# === Standard Libraries ===
import os
import numpy as np
from typing import Tuple, List, Optional

# === Third-Party Libraries ===
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Or 'Agg' if no GUI is needed
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


GRID_ROWS = 6
GRID_COLS = 8


def axis_equal_3d(ax: Axes) -> None:
    """Set equal aspect ratio for 3D plot."""
    extents = np.array([getattr(ax, f"get_{dim}lim")() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, f"set_{dim}lim")(ctr - r, ctr + r)


def setup_figure(title="Calibration Result") -> Tuple[plt.figure, plt.axes]:
    """Setup the figure for 3D plot."""
    # Turn on interactive mode
    plt.ion()
    
    # Create a figure
    fig = plt.figure(figsize=(20, 15), dpi=180)
    
    # Add a 3D axis
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=28)
    ax.set_xlabel("X", fontsize=20)
    ax.set_ylabel("Y", fontsize=20)
    ax.set_zlabel("Z", fontsize=20)
    ax.set_xlim(-2100, 2100)    # width = 4000 mm
    ax.set_ylim(-1800, 1500)    # hight = 2000 mm
    ax.set_zlim(-350, 2800)     # depth = 3000 mm
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(True)
    ax.view_init(elev=100, azim=90)
    
    # Equal aspect ratio
    axis_equal_3d(ax)

    # Tight layout
    plt.tight_layout()
    
    # Make the figure full screen
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
        
    return fig, ax


def show_figure(save=False, folder_path = "outputs"):
    """ Show the figure. """
    plt.legend()
    
    # Save the figure
    if save:
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(f"{folder_path}/calibration_result.svg", dpi=300, bbox_inches='tight')
    
    # Show the figure
    plt.show(block=True)
    
    # Close the figure
    plt.close()


def add_tv_screen_to_subplot(
    ax,
    R: np.ndarray,
    tvec: np.ndarray,
    monitor_mm: tuple,
    tv_object: bool = True,
    axis_length: float = 300,
    linewidth: float = 3.0,
    origin_color: str = "#000000",
    origin_size: float = 40,
    label: str = "TV",
) -> None:
    """
    Adds a TV screen, its pose axes, and its origin point to a 3D plot.

    Parameters
    ----------
    ax : matplotlib 3D axis
        The axis to plot onto.
    R : np.ndarray
        3x3 rotation matrix (local → world).
    tvec : np.ndarray
        3x1 or (3,) translation vector in world coordinates.
    monitor_mm : tuple
        Physical resolution of the screen in mm (width, height).
    axis_length : float
        Length of each axis for the pose.
    linewidth : float
        Line width for pose axes and screen border.
    origin_color : str or tuple
        Color of the origin dot.
    origin_size : float
        Size of the origin dot.
    label : str
        Label for the screen (used in legend).
    """
    # --- Draw the TV screen surface ---
    draw_tv_screen(ax, R=R, tvec=tvec, monitor_mm=monitor_mm, tv_object=tv_object, label=label)

    # --- Draw the TV's pose (X, Y, Z axes) ---
    draw_pose_axes(ax, R=R, tvec=tvec, length=axis_length, linewidth=linewidth)

    # --- Draw the TV's origin point (top-left corner of screen) ---
    draw_origin_point(ax, tvec=tvec, color=origin_color, size=origin_size)


def draw_tv_screen(
    ax,
    R: np.ndarray,
    tvec: np.ndarray,
    monitor_mm: tuple,
    tv_object: bool = False,
    label: str = "TV"
) -> None:
    """
    Plot a rotated TV screen in 3D space using its R|tvec.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axis to draw on.
    R : np.ndarray
        3x3 rotation matrix (board-to-world).
    tvec : np.ndarray
        3x1 translation vector in world coordinates.
    monitor_mm : tuple
        Physical resolution of the screen in mm (width, height).
    tv_object : bool
        Whether to draw the TV as a 3D object or just its outline.
    label : str
        Label for the screen in legend.
    """
    width_mm, height_mm = monitor_mm

    # # Define 4 screen corners in local board frame
    screen_local = np.array([
        [0,          0,          0],  # bottom-left
        [width_mm,   0,          0],  # bottom-right
        [width_mm,   height_mm,  0],  # top-right
        [0,          height_mm,  0],  # top-left
        [0,          0,          0]   # close the loop
    ]).T  # shape: (3, 5)
    
    screen_world = (R @ screen_local) + tvec.reshape(3, 1)

    if tv_object:
        # Build list of 4-tuples for Poly3DCollection
        verts = [list(zip(screen_world[0], screen_world[1], screen_world[2]))]

        board_patch = Poly3DCollection(
            verts,
            facecolors=[(1, 1, 1, 0.15)],
            edgecolor="k",
            linestyles='-',
            linewidth=1.5,
            label=label
        )
        ax.add_collection3d(board_patch)
    else:
        # Plot screen outline
        ax.plot(
            xs=screen_world[0],
            ys=screen_world[1],
            zs=screen_world[2],
            color="#000000DE",
            linestyle='-',
            linewidth=1.5,
            label=label
        )

    # Vertical grid lines (parallel to Y-axis)
    for i in range(1, GRID_COLS):
        x = i * (width_mm / GRID_COLS)
        pt1 = np.array([[x], [0], [0]])
        pt2 = np.array([[x], [height_mm], [0]])
        p1w = (R @ pt1) + tvec.reshape(3, 1)
        p2w = (R @ pt2) + tvec.reshape(3, 1)
        ax.plot(
            [p1w[0, 0], p2w[0, 0]],
            [p1w[1, 0], p2w[1, 0]],
            [p1w[2, 0], p2w[2, 0]],
            color='#246B6D',
            linestyle='-',
            linewidth=0.5
        )

    # Horizontal grid lines (parallel to X-axis)
    for j in range(1, GRID_ROWS):
        y = j * (height_mm / GRID_ROWS)
        pt1 = np.array([[0], [y], [0]])
        pt2 = np.array([[width_mm], [y], [0]])
        p1w = (R @ pt1) + tvec.reshape(3, 1)
        p2w = (R @ pt2) + tvec.reshape(3, 1)
        ax.plot(
            [p1w[0, 0], p2w[0, 0]],
            [p1w[1, 0], p2w[1, 0]],
            [p1w[2, 0], p2w[2, 0]],
            color='#246B6D',
            linestyle='-',
            linewidth=0.5
        )

    # Text labels
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            cx = (col + 0.5) * (width_mm / GRID_COLS)
            cy = (row + 0.5) * (height_mm / GRID_ROWS)
            pt = np.array([[cx], [cy], [0]])
            ptw = (R @ pt) + tvec.reshape(3, 1)

            label_txt = f"{GRID_COLS * row + col + 1}"
            ax.text(
                ptw[0, 0],
                ptw[1, 0],
                ptw[2, 0],
                label_txt,
                color="#000000",
                ha='center',
                va='center',
                fontsize=5
            )


def draw_pose_axes(
    ax,
    R: np.ndarray,
    tvec: np.ndarray,
    length: float = 200,
    length_multiplier: tuple = (1.0, 1.0, 1.0),
    linewidth: float = 3.0,
    label: str = None
):
    """
    Draw 3D coordinate axes at a given pose (R, t).

    Parameters
    ----------
    ax : matplotlib 3D axis
    R : np.ndarray
        3x3 rotation matrix (from local to world)
    tvec : np.ndarray
        3x1 or (3,) translation vector in world coordinates
    length : float
        Length of each axis in mm
    linewidth : float
        Thickness of the axis lines
    label : str
        Optional label for the pose (used only once)
    """
    origin = tvec.reshape(3, 1)
    xs, ys, zs = length_multiplier
    
    # Direction vectors for X, Y, Z in world frame
    x_axis = origin + R @ (np.array([[length*xs], [0], [0]]))
    y_axis = origin + R @ (np.array([[0], [length*ys], [0]]))
    z_axis = origin + R @ (np.array([[0], [0], [length*zs]]))

    # Draw the axes
    ax.plot(
        [origin[0, 0], x_axis[0, 0]],
        [origin[1, 0], x_axis[1, 0]],
        [origin[2, 0], x_axis[2, 0]],
        color='black',
        linewidth=linewidth
    )
    ax.plot(
        [origin[0, 0], y_axis[0, 0]],
        [origin[1, 0], y_axis[1, 0]],
        [origin[2, 0], y_axis[2, 0]],
        color='g',
        linewidth=linewidth
    )
    ax.plot(
        [origin[0, 0], z_axis[0, 0]],
        [origin[1, 0], z_axis[1, 0]],
        [origin[2, 0], z_axis[2, 0]],
        color='r',
        linewidth=linewidth
    )


def draw_origin_point(
    ax,
    tvec: np.ndarray,
    color: str = '#000000',
    size: float = 20,
    edgecolors: str = 'k',
    linewidths: float = 1.0,
    alpha: float = 0.9,
    label: str = None,
) -> None:
    """
    Draw a marker at the origin position defined by tvec.

    Parameters
    ----------
    ax : matplotlib 3D axis
        The axis to draw on.
    tvec : np.ndarray
        3x1 or (3,) array specifying the position in world coordinates.
    color : str or tuple
        Color of the point (RGB tuple or matplotlib string).
    size : float
        Size of the point.
    edgecolors : str or tuple
        Color of the point edges.
    linewidths : float
        Width of the point edges.
    alpha : float
        Transparency of the point.
    label : str
        Legend label for the point.
    """
    origin = tvec.reshape(3,)
    
    ax.scatter(
        origin[0],
        origin[1],
        origin[2],
        color=color,
        s=size,
        edgecolors=edgecolors,
        linewidths=linewidths,
        alpha=alpha,
        label=label
    )


def plot_camera_object(
    ax: Axes,
    coords: List[np.ndarray],
    color: tuple[float],
    linewidth: float,
) -> None:
    assert len(coords) == 12
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = coords

    ax.plot(
        xs=[x1[0][0], x2[0][0]],
        ys=[x1[1][0], x2[1][0]],
        zs=[x1[2][0], x2[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x1[0][0], x3[0][0]],
        ys=[x1[1][0], x3[1][0]],
        zs=[x1[2][0], x3[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x4[0][0], x3[0][0]],
        ys=[x4[1][0], x3[1][0]],
        zs=[x4[2][0], x3[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x2[0][0], x4[0][0]],
        ys=[x2[1][0], x4[1][0]],
        zs=[x2[2][0], x4[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x5[0][0], x6[0][0]],
        ys=[x5[1][0], x6[1][0]],
        zs=[x5[2][0], x6[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x5[0][0], x7[0][0]],
        ys=[x5[1][0], x7[1][0]],
        zs=[x5[2][0], x7[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x8[0][0], x7[0][0]],
        ys=[x8[1][0], x7[1][0]],
        zs=[x8[2][0], x7[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x6[0][0], x8[0][0]],
        ys=[x6[1][0], x8[1][0]],
        zs=[x6[2][0], x8[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x1[0][0], x5[0][0]],
        ys=[x1[1][0], x5[1][0]],
        zs=[x1[2][0], x5[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x2[0][0], x6[0][0]],
        ys=[x2[1][0], x6[1][0]],
        zs=[x2[2][0], x6[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x3[0][0], x7[0][0]],
        ys=[x3[1][0], x7[1][0]],
        zs=[x3[2][0], x7[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x4[0][0], x8[0][0]],
        ys=[x4[1][0], x8[1][0]],
        zs=[x4[2][0], x8[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x9[0][0], x10[0][0]],
        ys=[x9[1][0], x10[1][0]],
        zs=[x9[2][0], x10[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x9[0][0], x11[0][0]],
        ys=[x9[1][0], x11[1][0]],
        zs=[x9[2][0], x11[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x12[0][0], x11[0][0]],
        ys=[x12[1][0], x11[1][0]],
        zs=[x12[2][0], x11[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x10[0][0], x12[0][0]],
        ys=[x10[1][0], x12[1][0]],
        zs=[x10[2][0], x12[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x5[0][0], x9[0][0]],
        ys=[x5[1][0], x9[1][0]],
        zs=[x5[2][0], x9[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x6[0][0], x10[0][0]],
        ys=[x6[1][0], x10[1][0]],
        zs=[x6[2][0], x10[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x7[0][0], x11[0][0]],
        ys=[x7[1][0], x11[1][0]],
        zs=[x7[2][0], x11[2][0]],
        color=color,
        linewidth=linewidth,
    )
    ax.plot(
        xs=[x8[0][0], x12[0][0]],
        ys=[x8[1][0], x12[1][0]],
        zs=[x8[2][0], x12[2][0]],
        color=color,
        linewidth=linewidth,
    )


def add_camera_to_subplot(
    ax: Axes,
    R: np.ndarray,
    tvec: np.ndarray,
    cam_size: float = 0,
    cam_edge: float = 1.0,
    cam_color: Optional[List[float]] = None,
    label: Optional[str] = None,
) -> None:
    if not cam_color:
        cam_color = [0.0, 0.0, 0.0]

    r = cam_size
    r1_5 = r * 1.5 
    r2_0 = r * 2 
    r3_0 = r * 3 

    # Corners of the camera in the camera coordinate system
    x1 = np.asarray([[ r, -r, -r2_0]]).T
    x2 = np.asarray([[ r,  r, -r2_0]]).T
    x3 = np.asarray([[-r, -r, -r2_0]]).T
    x4 = np.asarray([[-r,  r, -r2_0]]).T

    x5 = np.asarray([[ r, -r, r2_0]]).T
    x6 = np.asarray([[ r,  r, r2_0]]).T
    x7 = np.asarray([[-r, -r, r2_0]]).T
    x8 = np.asarray([[-r,  r, r2_0]]).T

    x9 =  np.asarray([[ r1_5, -r1_5, r3_0]]).T
    x10 = np.asarray([[ r1_5,  r1_5, r3_0]]).T
    x11 = np.asarray([[-r1_5, -r1_5, r3_0]]).T
    x12 = np.asarray([[-r1_5,  r1_5, r3_0]]).T

    # Corners of the camera in the world coordinate system
    x1 = np.matmul(R, x1) + tvec
    x2 = np.matmul(R, x2) + tvec
    x3 = np.matmul(R, x3) + tvec
    x4 = np.matmul(R, x4) + tvec
    x5 = np.matmul(R, x5) + tvec
    x6 = np.matmul(R, x6) + tvec
    x7 = np.matmul(R, x7) + tvec
    x8 = np.matmul(R, x8) + tvec
    x9 = np.matmul(R, x9) + tvec
    x10 = np.matmul(R, x10) + tvec
    x11 = np.matmul(R, x11) + tvec
    x12 = np.matmul(R, x12) + tvec

    camera_coords = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12]
    
    plot_camera_object(ax, camera_coords, cam_color, cam_edge)
    draw_pose_axes(ax, R=R, tvec=tvec, length=180, length_multiplier=(0, 0, 1), linewidth=1.0)
    draw_origin_point(ax, tvec=tvec, color=cam_color, size=50, label=label)


def plot_target_on_screen(
    ax,
    point_px: Tuple[int, int],
    monitor_px: Tuple[int, int],
    monitor_mm: Tuple[float, float],
) -> np.ndarray:
    """
    Plot the target on the screen and return its 3D coordinates in the camera coordinate system.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on.
    point_px : tuple
        Pixel coordinates of the target on the screen.
    monitor_px : tuple
        Resolution of the screen in pixels.
    monitor_mm : tuple
        Resolution of the screen in mm.
    Returns
    -------
    np.ndarray
        3D coordinates of the target in the camera coordinate system.
    """
    px, py = point_px
    W_px, H_px = monitor_px # Monitor resolution in pixels
    W_mm, H_mm = monitor_mm # Monitor physical dimensions in mm

    # Convert pixel coordinates to millimeters in the screen coordinate system
    x_mm = px * (W_mm / W_px)   # Scale X coordinate
    y_mm = py * (H_mm / H_px)   # Scale Y coordinate
    z_mm = 0.0                  # Z=0 is the screen plane
    
    # Shift x coordinate to match screen coordinate system
    x_mm = (W_mm / 2) - x_mm
    
    # Create 3D point in screen coordinate system 
    screen_pt = np.array([[x_mm], [y_mm], [z_mm]], dtype=np.float32)
        
    ax.plot(
        x_mm,
        y_mm,
        z_mm,
        linestyle="",
        marker="X",
        color='#9467bd',
        label='target on screen'
    )
    return screen_pt.flatten()


def plot_face_landmarks(ax, landmarks_3D, inliers=None):
    if inliers is not None:
        N = landmarks_3D.shape[1]  # total number of points
        mask = np.zeros(N, dtype=bool)
        mask[inliers[:, 0]] = True

        landmarks_3D_inliers = landmarks_3D[:, mask]
        landmarks_3D_outliers   = landmarks_3D[:, ~mask]
    
        ax.plot(
            landmarks_3D_inliers[0, :],
            landmarks_3D_inliers[1, :],
            landmarks_3D_inliers[2, :],
            linestyle="",
            marker="o",
            color='#7f7f7f',
            markersize=1,
            label='face landmarks (inliers)'
        )
        ax.plot(
            landmarks_3D_outliers[0, :],
            landmarks_3D_outliers[1, :],
            landmarks_3D_outliers[2, :],
            linestyle="",
            marker="o",
            color="#fd0a0a",
            markersize=1,
            label='face landmarks (outliers)'
        )
    else:
        ax.plot(
            landmarks_3D[0, :],
            landmarks_3D[1, :],
            landmarks_3D[2, :],
            linestyle="",
            marker="o",
            color='#7f7f7f',
            markersize=1,
            label='face landmarks'
        )


def plot_gaze_vectors(ax, landmarks_3D, screen_pt):
    """
    Plot the gaze vectors of the right and left eye in 3D.

    The gaze vectors are plotted as lines from the target on the screen to the average of the right or left eye center points.

    :param ax: matplotlib Axes object
    :param landmarks_3D: (3, 468) numpy array of 3D positions of facial landmarks in the camera coordinate system
    :param screen_pt: (3,) numpy array of 3D position of the target on the screen
    """
    def valid_avg(p1, p2):
        if np.isnan(p1).any() or np.isnan(p2).any():
            return None
        return (p1 + p2) / 2

    reye = valid_avg(landmarks_3D[:, 33], landmarks_3D[:, 133])
    leye = valid_avg(landmarks_3D[:, 263], landmarks_3D[:, 362])

    if reye is not None:
        ax.plot(
            [screen_pt[0], reye[0]],
            [screen_pt[1], reye[1]],
            [screen_pt[2], reye[2]],
            color='#2ca02c', label='right eye gaze vector'
        )
    if leye is not None:
        ax.plot(
            [screen_pt[0], leye[0]],
            [screen_pt[1], leye[1]],
            [screen_pt[2], leye[2]],
            color='#d62728', label='left eye gaze vector'
        )
