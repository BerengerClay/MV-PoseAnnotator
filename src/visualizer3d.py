import numpy as np
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.constants import KEYPOINT_COLORS, COCO_SKELETON
from src.icons import get_lucide_icon, configure_button

class Visualizer3DWidget(QWidget):
    """Matplotlib-based 3D skeleton visualizer widget that can be used inline or inside a window."""
    def __init__(self, main_win, parent=None, small_mode=False):
        super().__init__(parent)
        self.main_win = main_win
        self.small_mode = small_mode
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Matplotlib Figure and Canvas
        self.figure = Figure(facecolor='#090d16')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # 3D Axes
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#090d16')
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Style grid/panes of 3D plot
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#1e293b')
        self.ax.yaxis.pane.set_edgecolor('#1e293b')
        self.ax.zaxis.pane.set_edgecolor('#1e293b')
        self.ax.grid(True, color='#334155', linestyle='--')
        
        if self.small_mode:
            self.ax.tick_params(colors='#94a3b8', labelsize=7, pad=1)
            
            # Tiny popout button in the corner of the small visualizer
            self.btn_popout = QPushButton(self)
            self.btn_popout.setIcon(get_lucide_icon("external-link", color="#38bdf8"))
            self.btn_popout.setStyleSheet("background-color: rgba(15, 23, 42, 200); border: 1px solid #334155; padding: 2px; border-radius: 4px;")
            self.btn_popout.clicked.connect(self.main_win.toggle_3d_window)
        else:
            self.ax.set_xlabel('X (m)', color='#94a3b8')
            self.ax.set_ylabel('Y (m)', color='#94a3b8')
            self.ax.set_zlabel('Z (m)', color='#94a3b8')
            self.ax.tick_params(colors='#94a3b8')
            
        self.view_mode = "athlete" # "athlete" or "global"
        self.update_plot(None)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'btn_popout') and self.btn_popout:
            self.btn_popout.resize(24, 24)
            self.btn_popout.move(self.width() - 28, 4)

    def update_plot(self, pts_3d):
        """Updates the 3D scatter and line plots with new 3D keypoint coordinates."""
        self.ax.cla()
        self.ax.set_facecolor('#090d16')
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.grid(True, color='#334155', linestyle='--')
        
        if self.small_mode:
            self.ax.tick_params(colors='#94a3b8', labelsize=7, pad=1)
        else:
            self.ax.set_xlabel('X (m)', color='#94a3b8')
            self.ax.set_ylabel('Y (m)', color='#94a3b8')
            self.ax.set_zlabel('Z (m)', color='#94a3b8')
            self.ax.tick_params(colors='#94a3b8')
            
        if pts_3d is None or np.all(np.isnan(pts_3d)):
            self.ax.text2D(0.5, 0.5, "Not enough points\n(triangulate to generate 3D)", 
                           color='#94a3b8', ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return
            
        # Draw skeleton lines with matching segment colors
        for conn in COCO_SKELETON:
            p1, p2 = conn
            pt1 = pts_3d[p1]
            pt2 = pts_3d[p2]
            
            if not np.isnan(pt1[0]) and not np.isnan(pt2[0]):
                if conn in [(5, 6), (11, 12)]:
                    # Torso/Trunk (Emerald Green)
                    col_str = '#10b981'
                elif conn in [(0, 1), (0, 2), (1, 3), (2, 4)]:
                    # Head/Face (Magenta/Pink)
                    col_str = '#ec4899'
                elif p1 in [5, 7, 9, 11, 13, 15] and p2 in [5, 7, 9, 11, 13, 15]:
                    # Left side (Cyan)
                    col_str = '#06b6d4'
                else:
                    # Right side (Orange/Red)
                    col_str = '#f97316'
                    
                self.ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 
                             color=col_str, linewidth=1.5 if self.small_mode else 2, zorder=2)

        # Draw joints
        xs, ys, zs = [], [], []
        colors = []
        
        for idx in range(17):
            pt = pts_3d[idx]
            if not np.isnan(pt[0]):
                xs.append(pt[0])
                ys.append(pt[1])
                zs.append(pt[2])
                qcol = KEYPOINT_COLORS.get(idx, QColor(0, 255, 0))
                colors.append([qcol.red()/255.0, qcol.green()/255.0, qcol.blue()/255.0])
                
        if xs:
            self.ax.scatter(xs, ys, zs, c=colors, s=30 if self.small_mode else 50, depthshade=True, zorder=10)
                             
        # Set equal aspect ratio for 3D plot
        if self.view_mode == "global" and self.main_win and getattr(self.main_win, 'global_3d_bounds', None) is not None:
            bounds = self.main_win.global_3d_bounds
            max_range = max(bounds["x_max"] - bounds["x_min"], 
                            bounds["y_max"] - bounds["y_min"], 
                            bounds["z_max"] - bounds["z_min"])
            if max_range == 0:
                max_range = 1.0
            mid_x = (bounds["x_max"] + bounds["x_min"]) * 0.5
            mid_y = (bounds["y_max"] + bounds["y_min"]) * 0.5
            mid_z = (bounds["z_max"] + bounds["z_min"]) * 0.5
            
            self.ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
            self.ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
            self.ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
        else:
            all_x = pts_3d[~np.isnan(pts_3d[:, 0]), 0]
            all_y = pts_3d[~np.isnan(pts_3d[:, 1]), 1]
            all_z = pts_3d[~np.isnan(pts_3d[:, 2]), 2]
            
            if len(all_x) > 0:
                max_range = max(all_x.max() - all_x.min(), 
                                all_y.max() - all_y.min(), 
                                all_z.max() - all_z.min())
                if max_range == 0:
                    max_range = 1.0
                mid_x = (all_x.max() + all_x.min()) * 0.5
                mid_y = (all_y.max() + all_y.min()) * 0.5
                mid_z = (all_z.max() + all_z.min()) * 0.5
                
                self.ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
                self.ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
                self.ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
            
        self.canvas.draw()


class Visualizer3DWindow(QMainWindow):
    """Separate window container for the 3D visualizer that adds play/pause, sync and progress controls."""
    def __init__(self, main_win, parent=None):
        super().__init__(parent)
        self.main_win = main_win
        self.setWindowTitle("3D Skeleton Visualizer")
        self.resize(600, 650)
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 3D Visualizer Canvas
        self.widget_3d = Visualizer3DWidget(main_win, self, small_mode=False)
        main_layout.addWidget(self.widget_3d, stretch=1)
        
        # Playback control panel at the bottom
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)
        
        self.btn_play_pause = QPushButton()
        configure_button(self.btn_play_pause, text="Play", icon_name="play", icon_color="#ffffff", bg_color="#059669")
        self.btn_play_pause.clicked.connect(self.toggle_playback)
        
        self.btn_sync = QPushButton()
        configure_button(self.btn_sync, text="Sync to Annotator", icon_name="refresh-cw", icon_color="#ffffff")
        self.btn_sync.clicked.connect(self.sync_to_annotator_frame)
        self.btn_sync.setToolTip("Return the visualizer frame to match the annotator's active frame")
        self.btn_sync.setStyleSheet("background-color: #1e293b; color: white; border: 1px solid #334155;")
        
        self.btn_view_mode = QPushButton()
        configure_button(self.btn_view_mode, text="Focus Mode", icon_name="maximize", icon_color="#ffffff")
        self.btn_view_mode.clicked.connect(self.toggle_view_mode)
        self.btn_view_mode.setToolTip("Toggle between Athlete Focus and Global View Mode")
        self.btn_view_mode.setStyleSheet("background-color: #1e293b; color: white; border: 1px solid #334155;")
        
        self.playback_slider = QSlider(Qt.Orientation.Horizontal)
        self.playback_slider.valueChanged.connect(self.on_slider_moved)
        self.playback_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #475569;
                height: 6px;
                background: #1e293b;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #38bdf8;
                border: 1px solid #0284c7;
                width: 14px;
                height: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
        """)
        
        self.lbl_frame_info = QLabel("Frame: 0 / 0")
        self.lbl_frame_info.setStyleSheet("color: #94a3b8; font-weight: bold; min-width: 90px;")
        
        control_layout.addWidget(self.btn_play_pause)
        control_layout.addWidget(self.btn_sync)
        control_layout.addWidget(self.btn_view_mode)
        control_layout.addWidget(self.playback_slider)
        control_layout.addWidget(self.lbl_frame_info)
        
        main_layout.addWidget(control_panel)
        
        # Playback timer
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.advance_frame)
        self.playback_frame_idx = 0
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
            }
            QWidget {
                background-color: #0f172a;
                color: #f8fafc;
            }
            QPushButton {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #334155;
            }
        """)

    def toggle_playback(self):
        if not self.main_win or not self.main_win.sorted_frames:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            configure_button(self.btn_play_pause, text="Play", icon_name="play", icon_color="#ffffff", bg_color="#059669")
        else:
            self.play_timer.start(100) # 10 FPS
            configure_button(self.btn_play_pause, text="Pause", icon_name="pause", icon_color="#ffffff", bg_color="#d97706")

    def advance_frame(self):
        if not self.main_win or not self.main_win.sorted_frames:
            self.play_timer.stop()
            return
        total = len(self.main_win.sorted_frames)
        self.playback_frame_idx = (self.playback_frame_idx + 1) % total
        
        self.playback_slider.blockSignals(True)
        self.playback_slider.setValue(self.playback_frame_idx)
        self.playback_slider.blockSignals(False)
        
        self.lbl_frame_info.setText(f"Frame: {self.playback_frame_idx + 1} / {total}")
        self.update_visualization()

    def on_slider_moved(self, value):
        self.playback_frame_idx = value
        total = len(self.main_win.sorted_frames) if (self.main_win and self.main_win.sorted_frames) else 0
        self.lbl_frame_info.setText(f"Frame: {self.playback_frame_idx + 1} / {total}")
        self.update_visualization()

    def sync_to_annotator_frame(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            configure_button(self.btn_play_pause, text="Play", icon_name="play", icon_color="#ffffff", bg_color="#059669")
        if self.main_win and self.main_win.sorted_frames:
            self.playback_frame_idx = self.main_win.current_frame_idx
            
            self.playback_slider.blockSignals(True)
            self.playback_slider.setValue(self.playback_frame_idx)
            self.playback_slider.blockSignals(False)
            
            total = len(self.main_win.sorted_frames)
            self.lbl_frame_info.setText(f"Frame: {self.playback_frame_idx + 1} / {total}")
        self.update_visualization()

    def toggle_view_mode(self):
        if self.widget_3d.view_mode == "athlete":
            self.widget_3d.view_mode = "global"
            configure_button(self.btn_view_mode, text="Global Mode", icon_name="globe", icon_color="#ffffff")
        else:
            self.widget_3d.view_mode = "athlete"
            configure_button(self.btn_view_mode, text="Focus Mode", icon_name="maximize", icon_color="#ffffff")
        self.update_visualization()

    def update_visualization(self):
        if self.main_win:
            pts_3d = self.main_win.calculate_3d_keypoints(self.playback_frame_idx)
            self.widget_3d.update_plot(pts_3d)

    def closeEvent(self, event):
        self.play_timer.stop()
        super().closeEvent(event)
