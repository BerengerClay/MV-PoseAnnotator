import os
import re
import json
import cv2
import tomllib
import numpy as np
import torch

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QMessageBox,
    QLabel,
    QStatusBar,
    QProgressDialog,
    QSlider,
    QDialog,
    QSpinBox,
)
from PyQt6.QtGui import QKeySequence, QShortcut, QColor
from PyQt6.QtCore import Qt, QTimer

from src.constants import CAMERA_KEYS
from src.widgets import CameraWidget
from src.workers import WorkerThread, SequencePreprocessWorker
from src.dialogs import (
    SettingsDialog,
    SelectCameraFoldersDialog,
    select_multiple_directories,
)
from src.visualizer3d import Visualizer3DWindow, Visualizer3DWidget
from src.backend import ModelWrapper
from src.icons import get_lucide_icon


SETTINGS_FILE = os.path.join("configs", "local_settings.json")


def log_debug(msg):
    try:
        import datetime

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open("annotator.log", "a", encoding="utf-8") as f:
            f.write(f"[{now}] {msg}\n")
            f.flush()
    except Exception:
        pass


class TrampolineAnnotator(QMainWindow):
    def __init__(self, paths=None):
        log_debug("TrampolineAnnotator.__init__ started")
        super().__init__()
        self.setWindowTitle("Multi-View Trampoline Jumper Annotator")
        self.setGeometry(100, 100, 1600, 950)

        # Application state
        self.sequence_dir = None
        self.camera_dirs = {}
        self.json_path = None
        self.sorted_frames = []
        self.current_frame_idx = -1
        self.frame_data = {}  # frame_idx -> {camera_key -> filepath}

        self.coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "person"}],
        }
        self.img_ann_map = {}  # image_id -> annotation dict
        self.img_file_map = {}  # file_name -> image dict

        # Load camera matrices
        self.camera_matrices = self.load_camera_matrices()
        self.calib_data = self.load_calib_data()

        # Load local settings
        saved_settings = self.load_local_settings() or {}

        # Deep learning models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resolve paths from local settings or use default paths
        src_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(src_dir)
        
        default_yolo_path = os.path.join(root_dir, "weights", "YOLO26s_best.pt")
        default_vitpose_path = os.path.join(root_dir, "weights", "best_ViTPose-s_AP731.pth")
        
        def resolve_path(p, default):
            if not p:
                return default
            if os.path.isabs(p):
                return p
            return os.path.abspath(os.path.join(root_dir, p))

        self.yolo_path = resolve_path(saved_settings.get("yolo_path"), default_yolo_path)
        self.vitpose_path = resolve_path(saved_settings.get("vitpose_path"), default_vitpose_path)

        self.model_wrapper = ModelWrapper(
            weights_dir=None,
            device=self.device,
            yolo_path=self.yolo_path,
            vitpose_path=self.vitpose_path
        )
        self.active_worker = None
        self.keypoint_radius = saved_settings.get("keypoint_radius", 3)
        self.visualizer_3d_window = None
        self.auto_rotate_enabled = saved_settings.get("auto_rotate_enabled", True)
        self.global_3d_bounds = None
        self.show_3d_reprojection = saved_settings.get("show_3d_reprojection", False)
        self.realtime_triangulation_enabled = saved_settings.get(
            "realtime_triangulation_enabled", False
        )
        self.delete_bbox_on_clear = saved_settings.get("delete_bbox_on_clear", False)
        self.vitpose_show_confidence = saved_settings.get(
            "vitpose_show_confidence", True
        )
        self.vitpose_threshold = saved_settings.get("vitpose_threshold", 0.2)

        # History stacks for Undo/Redo
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 50

        self.init_ui()
        self.apply_dark_style()
        self.setup_shortcuts()

        # Load sequence if provided via argument, otherwise check saved session, else prompt
        if paths:
            self.load_sequence_from_cli_paths(paths)
        else:
            saved_dirs = saved_settings.get("camera_dirs")
            dirs_valid = False
            resolved_dirs = {}
            if isinstance(saved_dirs, dict) and len(saved_dirs) == len(CAMERA_KEYS):
                project_root = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
                dirs_valid = True
                for k, path in saved_dirs.items():
                    if os.path.isdir(path):
                        resolved_dirs[k] = os.path.abspath(path)
                    else:
                        full_path = os.path.abspath(os.path.join(project_root, path))
                        if os.path.isdir(full_path):
                            resolved_dirs[k] = full_path
                        else:
                            dirs_valid = False
                            break

            if dirs_valid:
                self.camera_dirs = resolved_dirs
                first_dir = next(iter(self.camera_dirs.values()))
                self.sequence_dir = os.path.dirname(first_dir)
                self.load_sequence_from_dirs(self.camera_dirs)
            else:
                self.prompt_select_sequence()
        log_debug("TrampolineAnnotator.__init__ completed successfully")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Left Panel: Camera views in a grid
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(10)
        self.camera_widgets = []

        # Set equal stretches to ensure the grid is perfectly balanced when restored
        for r in range(2):
            self.grid_layout.setRowStretch(r, 1)
        for c in range(4):
            self.grid_layout.setColumnStretch(c, 1)

        for i, key in enumerate(CAMERA_KEYS):
            cam = CameraWidget(camera_id=i, camera_name=key, main_win=self)
            self.camera_widgets.append(cam)
            self.grid_layout.addWidget(cam, i // 4, i % 4)

        main_layout.addLayout(self.grid_layout, stretch=4)

        # Right Panel: Sidebar control dashboard
        sidebar = QVBoxLayout()
        sidebar.setSpacing(15)

        # Sequence path label
        self.path_lbl = QLabel("No Sequence Loaded")
        self.path_lbl.setWordWrap(True)
        self.path_lbl.setStyleSheet("color: #64748b; font-size: 11px;")

        # Frame tracker row layout (label + undo/redo buttons)
        frame_row_layout = QHBoxLayout()

        self.frame_lbl = QLabel("Frame: 0 / 0")
        self.frame_lbl.setStyleSheet(
            "color: #f8fafc; font-size: 18px; font-weight: bold;"
        )
        frame_row_layout.addWidget(self.frame_lbl)
        frame_row_layout.addStretch()

        # Undo button next to label
        self.btn_undo = QPushButton()
        self.btn_undo.setIcon(get_lucide_icon("undo", color="#ffffff"))
        self.btn_undo.setToolTip("Undo last action (Ctrl+Z)")
        self.btn_undo.clicked.connect(self.undo)
        self.btn_undo.setEnabled(False)
        self.btn_undo.setFixedSize(28, 28)
        self.btn_undo.setStyleSheet("""
            QPushButton {
                padding: 4px;
                border-radius: 4px;
                background-color: #1e293b;
                border: 1px solid #334155;
            }
            QPushButton:hover {
                background-color: #334155;
                border-color: #475569;
            }
            QPushButton:disabled {
                background-color: transparent;
                border-color: transparent;
            }
        """)

        # Redo button next to label
        self.btn_redo = QPushButton()
        self.btn_redo.setIcon(get_lucide_icon("redo", color="#ffffff"))
        self.btn_redo.setToolTip("Redo last action (Ctrl+Shift+Z)")
        self.btn_redo.clicked.connect(self.redo)
        self.btn_redo.setEnabled(False)
        self.btn_redo.setFixedSize(28, 28)
        self.btn_redo.setStyleSheet("""
            QPushButton {
                padding: 4px;
                border-radius: 4px;
                background-color: #1e293b;
                border: 1px solid #334155;
            }
            QPushButton:hover {
                background-color: #334155;
                border-color: #475569;
            }
            QPushButton:disabled {
                background-color: transparent;
                border-color: transparent;
            }
        """)

        frame_row_layout.addWidget(self.btn_undo)
        frame_row_layout.addWidget(self.btn_redo)

        # Buttons
        btn_open = QPushButton("Select Sequence...")
        btn_open.setIcon(get_lucide_icon("folder-open", color="#f8fafc"))
        btn_open.clicked.connect(self.prompt_select_sequence)

        # Settings button (full width)
        self.btn_settings = QPushButton("Settings")
        self.btn_settings.setIcon(get_lucide_icon("settings", color="#f8fafc"))
        self.btn_settings.setToolTip("Application Settings")
        self.btn_settings.clicked.connect(self.show_settings)
        self.btn_settings.setStyleSheet(
            "background-color: #1e293b; border: 1px solid #334155;"
        )

        # Maximize view indicators
        self.mode_lbl = QLabel("Grid Mode (Double click view to zoom)")
        self.mode_lbl.setStyleSheet("color: #38bdf8; font-weight: bold;")

        # View navigation buttons (only shown in maximized view mode)
        self.view_nav_layout = QHBoxLayout()
        self.view_nav_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_prev_view = QPushButton("Prev View")
        self.btn_prev_view.setIcon(get_lucide_icon("arrow-left", color="#ffffff"))
        self.btn_prev_view.clicked.connect(self.show_prev_camera_view)
        self.btn_prev_view.setStyleSheet(
            "background-color: #1e293b; border: 1px solid #334155; padding: 6px 12px;"
        )

        self.btn_next_view = QPushButton("Next View")
        self.btn_next_view.setIcon(get_lucide_icon("arrow-right", color="#ffffff"))
        self.btn_next_view.clicked.connect(self.show_next_camera_view)
        self.btn_next_view.setStyleSheet(
            "background-color: #1e293b; border: 1px solid #334155; padding: 6px 12px;"
        )

        self.view_nav_layout.addWidget(self.btn_prev_view)
        self.view_nav_layout.addWidget(self.btn_next_view)

        self.btn_prev_view.hide()
        self.btn_next_view.hide()

        # AI commands and Triangulation

        self.btn_preprocess_seq = QPushButton("Preprocess Sequence")
        self.btn_preprocess_seq.setIcon(get_lucide_icon("sparkles", color="#ffffff"))
        self.btn_preprocess_seq.clicked.connect(self.run_sequence_preprocessing)
        self.btn_preprocess_seq.setEnabled(False)
        self.btn_preprocess_seq.setStyleSheet("""
            QPushButton {
                background-color: #7c3aed;
                color: white;
                font-weight: bold;
                border: 1px solid #7c3aed;
            }
            QPushButton:hover {
                background-color: #8b5cf6;
                border-color: #8b5cf6;
            }
            QPushButton:pressed {
                background-color: #6d28d9;
                border-color: #6d28d9;
            }
            QPushButton:disabled {
                background-color: #1e1b4b;
                color: #64748b;
                border-color: #1e1b4b;
            }
        """)
        self.btn_preprocess_seq.setToolTip(
            "Run YOLO + ViTPose from current frame to the end of the sequence"
        )

        self.btn_clear_frame_ann = QPushButton("Clear Frame")
        self.btn_clear_frame_ann.setIcon(get_lucide_icon("trash-2", color="#ffffff"))
        self.btn_clear_frame_ann.clicked.connect(self.clear_current_frame_annotations)
        self.btn_clear_frame_ann.setEnabled(False)
        self.btn_clear_frame_ann.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: white;
                font-weight: bold;
                border: 1px solid #dc2626;
            }
            QPushButton:hover {
                background-color: #ef4444;
                border-color: #ef4444;
            }
            QPushButton:pressed {
                background-color: #b91c1c;
                border-color: #b91c1c;
            }
            QPushButton:disabled {
                background-color: #451a03;
                color: #64748b;
                border-color: #451a03;
            }
        """)
        self.btn_clear_frame_ann.setToolTip(
            "Clear annotations for the current frame across all cameras"
        )

        self.ai_buttons_layout = QHBoxLayout()
        self.ai_buttons_layout.setSpacing(10)
        self.ai_buttons_layout.addWidget(self.btn_preprocess_seq)
        self.ai_buttons_layout.addWidget(self.btn_clear_frame_ann)

        self.btn_zoom_all = QPushButton("Zoom 8 Views to BBox")
        self.btn_zoom_all.setIcon(get_lucide_icon("maximize-2", color="#ffffff"))
        self.btn_zoom_all.clicked.connect(self.zoom_all_bboxes)
        self.btn_zoom_all.setEnabled(False)
        self.btn_zoom_all.setStyleSheet("background-color: #0369a1; color: white;")
        self.btn_zoom_all.setToolTip(
            "Zoom and rotate all 8 camera views onto their bounding boxes"
        )

        # Navigation slider
        self.slider_frame = QSlider(Qt.Orientation.Horizontal)
        self.slider_frame.setRange(0, 0)
        self.slider_frame.setValue(0)
        self.slider_frame.setEnabled(False)
        self.slider_frame.valueChanged.connect(self.on_slider_frame_changed)
        self.slider_frame.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #475569;
                height: 8px;
                background: #1e293b;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #38bdf8;
                border: 1px solid #0284c7;
                width: 16px;
                height: 16px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 8px;
            }
        """)

        # Spinbox for frame editing
        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(1, 1)
        self.spin_frame.setValue(1)
        self.spin_frame.setEnabled(False)
        self.spin_frame.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.spin_frame.valueChanged.connect(self.on_spin_frame_changed)
        self.spin_frame.setStyleSheet("""
            QSpinBox {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 2px 4px;
                font-weight: bold;
                font-size: 11px;
                min-width: 45px;
                max-width: 60px;
            }
        """)

        # Total frames label next to spinbox
        self.lbl_total_frames = QLabel("/ 0")
        self.lbl_total_frames.setStyleSheet(
            "color: #94a3b8; font-size: 11px; font-weight: bold;"
        )

        # Navigation
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("Previous")
        self.btn_prev.setIcon(get_lucide_icon("arrow-left", color="#ffffff"))
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_next = QPushButton("Next")
        self.btn_next.setIcon(get_lucide_icon("arrow-right", color="#ffffff"))
        self.btn_next.clicked.connect(self.next_frame)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)

        # Bottom controls container
        bottom_nav_layout = QVBoxLayout()
        bottom_nav_layout.addLayout(nav_layout)

        slider_row_layout = QHBoxLayout()
        slider_row_layout.addWidget(self.slider_frame, stretch=4)
        slider_row_layout.addWidget(self.spin_frame, stretch=1)
        slider_row_layout.addWidget(self.lbl_total_frames)
        bottom_nav_layout.addLayout(slider_row_layout)

        # Real-time inline 3D Visualizer widget
        self.visualizer_3d_inline = Visualizer3DWidget(self, small_mode=True)
        self.visualizer_3d_inline.setMinimumHeight(240)

        # Assembly
        sidebar.addLayout(frame_row_layout)
        sidebar.addWidget(self.path_lbl)
        sidebar.addWidget(btn_open)
        sidebar.addWidget(self.btn_settings)
        sidebar.addWidget(self.mode_lbl)
        sidebar.addLayout(self.view_nav_layout)
        sidebar.addSpacing(15)
        sidebar.addLayout(self.ai_buttons_layout)
        sidebar.addWidget(self.btn_zoom_all)
        sidebar.addSpacing(15)

        sidebar.addStretch()  # Push everything below to the bottom!

        # Bottom area: Inline 3D visualizer
        sidebar.addWidget(self.visualizer_3d_inline)
        sidebar.addSpacing(15)

        # Navigation buttons and slider at the very bottom
        sidebar.addLayout(bottom_nav_layout)

        main_layout.addLayout(sidebar, stretch=1)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def load_camera_matrices(self):
        """Loads matrices mapping 3D coordinates to 2D pixel coordinates."""
        src_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(src_dir)
        path = os.path.join(root_dir, "configs", "camera_matrices.json")
        if not os.path.exists(path):
            path = "configs/camera_matrices.json"
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load camera matrices: {e}")
            return {}

    def load_calib_data(self):
        """Loads camera calibration containing lens distortion parameters from Calib.toml."""
        src_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(src_dir)
        path = os.path.join(root_dir, "configs", "Calib.toml")
        if not os.path.exists(path):
            path = "configs/Calib.toml"
        try:
            with open(path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            print(f"Could not load Calib.toml calibration parameters: {e}")
            return {}

    def setup_shortcuts(self):
        """Registers global hotkeys to accelerate annotations."""
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self.prev_frame)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self.next_frame)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.reset_camera_grid)
        QShortcut(QKeySequence(Qt.Key.Key_Y), self, self.trigger_yolo_vitpose)
        QShortcut(QKeySequence(Qt.Key.Key_S), self, self.save_annotations)

        # Undo/Redo keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self.redo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.redo)

    def push_undo(self):
        """Saves a deep copy of the current annotations to the undo stack and clears the redo stack."""
        import copy

        state = copy.deepcopy(self.coco_data.get("annotations", []))
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.update_history_actions_state()

    def undo(self):
        """Reverts to the last saved state in the undo stack."""
        if not self.undo_stack:
            return
        import copy

        current_state = copy.deepcopy(self.coco_data.get("annotations", []))
        self.redo_stack.append(current_state)

        previous_state = self.undo_stack.pop()
        self.coco_data["annotations"] = previous_state

        # Re-build annotation map
        self.img_ann_map.clear()
        for ann in self.coco_data["annotations"]:
            self.img_ann_map[ann["image_id"]] = ann

        # Refresh UI and save
        self.show_current_frame(preserve_view=True)
        self.global_3d_bounds = self.calculate_global_3d_bounds()
        self.update_3d_view()
        self.save_annotations()
        self.update_history_actions_state()

    def redo(self):
        """Restores the last undone state in the redo stack."""
        if not self.redo_stack:
            return
        import copy

        current_state = copy.deepcopy(self.coco_data.get("annotations", []))
        self.undo_stack.append(current_state)

        next_state = self.redo_stack.pop()
        self.coco_data["annotations"] = next_state

        # Re-build annotation map
        self.img_ann_map.clear()
        for ann in self.coco_data["annotations"]:
            self.img_ann_map[ann["image_id"]] = ann

        # Refresh UI and save
        self.show_current_frame(preserve_view=True)
        self.global_3d_bounds = self.calculate_global_3d_bounds()
        self.update_3d_view()
        self.save_annotations()
        self.update_history_actions_state()

    def update_history_actions_state(self):
        """Enables/disables undo and redo buttons based on stack state."""
        if hasattr(self, "btn_undo") and self.btn_undo:
            self.btn_undo.setEnabled(len(self.undo_stack) > 0)
        if hasattr(self, "btn_redo") and self.btn_redo:
            self.btn_redo.setEnabled(len(self.redo_stack) > 0)

    def apply_dark_style(self):
        """Applies a premium, HSL tailored dark QSS style."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #090d16;
            }
            QWidget {
                color: #f8fafc;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QDialog, QMessageBox, QFileDialog {
                background-color: #0f172a;
            }
            QDialog QLabel, QMessageBox QLabel, QFileDialog QLabel {
                color: #f8fafc;
            }
            QDialog QLineEdit, QFileDialog QLineEdit {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px;
            }
            QDialog QListView, QDialog QTreeView, QFileDialog QListView, QFileDialog QTreeView {
                background-color: #090d16;
                color: #f8fafc;
                border: 1px solid #334155;
            }
            QDialog QHeaderView::section, QFileDialog QHeaderView::section {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
            }
            QDialog QComboBox, QFileDialog QComboBox {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px;
            }
            QDialog QComboBox QAbstractItemView, QFileDialog QComboBox QAbstractItemView {
                background-color: #1e293b;
                color: #f8fafc;
            }
            QPushButton {
                background-color: #1e293b;
                border: 1px solid #334155;
                padding: 10px 15px;
                border-radius: 6px;
                font-weight: 500;
                font-size: 13px;
                color: #f8fafc;
            }
            QPushButton:hover {
                background-color: #334155;
                border-color: #475569;
            }
            QPushButton:pressed {
                background-color: #0f172a;
            }
            QPushButton:disabled {
                color: #475569;
                background-color: #0f172a;
                border-color: #1e293b;
            }
            QLabel {
                font-size: 13px;
            }
            QStatusBar {
                background-color: #0f172a;
                color: #94a3b8;
                border-top: 1px solid #1e293b;
            }
            QToolTip {
                background-color: #0f172a;
                color: #f8fafc;
                border: 1px solid #334155;
                padding: 5px;
                border-radius: 4px;
            }
        """)

    def prompt_select_sequence(self):
        """Open a single file dialog to select multiple camera directories."""
        initial_dir = "/usagers4/p123652/Documents/annotator/Data"
        if not os.path.exists(initial_dir):
            initial_dir = "/usagers4/p123652/Documents/annotator"
        if not os.path.exists(initial_dir):
            initial_dir = os.path.expanduser("~")

        selected_paths = select_multiple_directories(
            self, "Select 8 Camera Folders", initial_dir
        )
        if not selected_paths:
            self.status_bar.showMessage("Sequence loading cancelled.")
            return

        # Attempt to map selected folders (could be 1 or more) to the 8 camera keys
        matched = {}
        unmatched = list(selected_paths)

        # Pass 1: exact or clean substring matches
        for key in CAMERA_KEYS:
            for path in list(unmatched):
                basename = os.path.basename(path).lower()
                key_clean = key.lower().replace("_", "").replace("-", "")
                base_clean = basename.replace("_", "").replace("-", "")
                if key.lower() in basename or key_clean in base_clean:
                    matched[key] = path
                    unmatched.remove(path)
                    break

        # Pass 2: map by camera number index
        for key in CAMERA_KEYS:
            if key in matched:
                continue
            match_cam_num = re.search(r"camera(\d+)", key.lower())
            if match_cam_num:
                num = match_cam_num.group(1)
                for path in list(unmatched):
                    basename = os.path.basename(path).lower()
                    if (
                        f"cam{num}" in basename
                        or f"camera{num}" in basename
                        or f"camera_{num}" in basename
                        or f"cam_{num}" in basename
                    ):
                        matched[key] = path
                        unmatched.remove(path)
                        break

        # If we matched all 8 cameras, we can load directly!
        if len(matched) == len(CAMERA_KEYS):
            all_valid = True
            for path in matched.values():
                try:
                    files = os.listdir(path)
                    if not any(
                        f.lower().endswith((".png", ".jpg", ".jpeg")) for f in files
                    ):
                        all_valid = False
                        break
                except Exception:
                    all_valid = False
                    break
            if all_valid:
                self.camera_dirs = matched
                first_dir = next(iter(self.camera_dirs.values()))
                self.sequence_dir = os.path.dirname(first_dir)
                self.load_sequence_from_dirs(self.camera_dirs)
                return

        # If some matched (but not all) or some are invalid, open the dialog pre-filled!
        first_dir = selected_paths[0]
        parent_est = os.path.dirname(first_dir)
        dialog = SelectCameraFoldersDialog(
            CAMERA_KEYS, initial_parent=parent_est, prefilled_dirs=matched, parent=self
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.camera_dirs = dialog.camera_dirs
            first_dir = next(iter(self.camera_dirs.values()))
            self.sequence_dir = os.path.dirname(first_dir)
            self.load_sequence_from_dirs(self.camera_dirs)

    def load_sequence_from_cli_paths(self, paths):
        """Loads sequence directly from a list of folders (multiple camera folders) passed via CLI."""
        if not paths:
            self.prompt_select_sequence()
            return

        matched = {}
        unmatched = list(paths)

        # Pass 1: exact or clean substring matches
        for key in CAMERA_KEYS:
            for path in list(unmatched):
                basename = os.path.basename(path).lower()
                key_clean = key.lower().replace("_", "").replace("-", "")
                base_clean = basename.replace("_", "").replace("-", "")
                if key.lower() in basename or key_clean in base_clean:
                    matched[key] = path
                    unmatched.remove(path)
                    break

        # Pass 2: map by camera number index
        for key in CAMERA_KEYS:
            if key in matched:
                continue
            match_cam_num = re.search(r"camera(\d+)", key.lower())
            if match_cam_num:
                num = match_cam_num.group(1)
                for path in list(unmatched):
                    basename = os.path.basename(path).lower()
                    if (
                        f"cam{num}" in basename
                        or f"camera{num}" in basename
                        or f"camera_{num}" in basename
                        or f"cam_{num}" in basename
                    ):
                        matched[key] = path
                        unmatched.remove(path)
                        break

        # If we successfully matched all 8 cameras, we load directly!
        if len(matched) == len(CAMERA_KEYS):
            all_valid = True
            for path in matched.values():
                try:
                    files = os.listdir(path)
                    if not any(
                        f.lower().endswith((".png", ".jpg", ".jpeg")) for f in files
                    ):
                        all_valid = False
                        break
                except Exception:
                    all_valid = False
                    break
            if all_valid:
                self.camera_dirs = matched
                first_dir = next(iter(self.camera_dirs.values()))
                self.sequence_dir = os.path.dirname(first_dir)
                self.load_sequence_from_dirs(self.camera_dirs)
                return

        # If not all were matched or valid, show warning and return without loading
        QMessageBox.warning(
            self,
            "Invalid Command Line Arguments",
            "Please specify all 8 camera folders when launching via command line.\n\n"
            "Example:\npython main.py Data/1_partie_0429_003*",
        )
        self.status_bar.showMessage(
            "Failed to load sequence: invalid command line arguments."
        )

    def extract_frame_idx(self, filename):
        """Extracts frame index from filename robustly."""
        match = re.search(r"frame_(\d+)", filename)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)", filename)
        if match:
            return int(match.group(1))
        return None

    def extract_video_id(self, paths):
        """Extracts the video identifier (e.g. '003' or '006') from path names."""
        for path in paths:
            basename = os.path.basename(path)
            match = re.search(r"_(\d+)-Camera", basename)
            if match:
                return match.group(1)
            match = re.search(r"_(\d+)-", basename)
            if match:
                return match.group(1)
            match = re.search(r"_(\d{3,})", basename)
            if match:
                return match.group(1)
        return "000"

    def load_sequence_from_dirs(self, camera_dirs):
        """Scans separate camera directories and loads or initializes annotation_{video_id}.json in parent's GT/ directory."""
        log_debug(f"load_sequence_from_dirs started, camera_dirs={camera_dirs}")
        # Convert all paths to absolute paths for consistency
        camera_dirs = {k: os.path.abspath(v) for k, v in camera_dirs.items()}
        self.camera_dirs = camera_dirs

        self.undo_stack.clear()
        self.redo_stack.clear()
        self.update_history_actions_state()

        first_dir = next(iter(camera_dirs.values()))
        self.sequence_dir = os.path.dirname(first_dir)

        self.path_lbl.setText(self.sequence_dir)

        gt_dir = os.path.join(self.sequence_dir, "GT")
        os.makedirs(gt_dir, exist_ok=True)

        video_id = self.extract_video_id(camera_dirs.values())
        self.json_path = os.path.join(gt_dir, f"annotation_{video_id}.json")

        self.status_bar.showMessage("Scanning camera directories...")
        self.frame_data.clear()

        for cam_key, cam_dir in camera_dirs.items():
            if not os.path.isdir(cam_dir):
                continue
            try:
                files = os.listdir(cam_dir)
            except Exception as e:
                print(f"Error listing {cam_dir}: {e}")
                continue
            for f in files:
                if not (
                    f.lower().endswith(".png")
                    or f.lower().endswith(".jpg")
                    or f.lower().endswith(".jpeg")
                ):
                    continue

                frame_idx = self.extract_frame_idx(f)
                if frame_idx is None:
                    continue

                if frame_idx not in self.frame_data:
                    self.frame_data[frame_idx] = {}
                self.frame_data[frame_idx][cam_key] = os.path.join(cam_dir, f)

        # Only keep frames that are present for all cameras to prevent KeyError later
        self.frame_data = {
            idx: cams
            for idx, cams in self.frame_data.items()
            if len(cams) == len(CAMERA_KEYS)
        }
        self.sorted_frames = sorted(list(self.frame_data.keys()))

        if not self.sorted_frames:
            QMessageBox.warning(
                self,
                "No frames found",
                "No valid camera frames found inside the camera directories.",
            )
            return

        self.coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "person"}],
        }
        self.img_ann_map.clear()
        self.img_file_map.clear()

        if os.path.exists(self.json_path):
            self.status_bar.showMessage(
                f"Loading existing {os.path.basename(self.json_path)}..."
            )
            try:
                with open(self.json_path, "r") as f:
                    self.coco_data = json.load(f)

                def get_path_key(path):
                    parts = os.path.normpath(path).split(os.sep)
                    if len(parts) >= 2:
                        return f"{parts[-2]}/{parts[-1]}"
                    return os.path.basename(path)

                existing_ann = {}
                existing_img = {}
                for img in self.coco_data.get("images", []):
                    existing_img[get_path_key(img["file_name"])] = img
                for ann in self.coco_data.get("annotations", []):
                    existing_ann[ann["image_id"]] = ann

                new_images = []
                new_annotations = []
                next_img_id = 1
                next_ann_id = 1

                if self.coco_data.get("images"):
                    next_img_id = max(img["id"] for img in self.coco_data["images"]) + 1
                if self.coco_data.get("annotations"):
                    next_ann_id = (
                        max(ann["id"] for ann in self.coco_data["annotations"]) + 1
                    )

                for frame_idx in self.sorted_frames:
                    for cam_key in CAMERA_KEYS:
                        if cam_key in self.frame_data[frame_idx]:
                            local_path = self.frame_data[frame_idx][cam_key]
                            path_key = get_path_key(local_path)

                            if path_key in existing_img:
                                img_entry = existing_img[path_key]
                                img_entry["file_name"] = local_path
                                ann_entry = existing_ann.get(img_entry["id"])
                                if ann_entry is None:
                                    ann_entry = {
                                        "id": next_ann_id,
                                        "image_id": img_entry["id"],
                                        "category_id": 1,
                                        "bbox": [0, 0, 0, 0],
                                        "keypoints": [0] * 51,
                                        "num_keypoints": 0,
                                        "iscrowd": 0,
                                    }
                                    next_ann_id += 1
                            else:
                                img_entry = {
                                    "id": next_img_id,
                                    "file_name": local_path,
                                    "width": 1920,
                                    "height": 1080,
                                }
                                ann_entry = {
                                    "id": next_ann_id,
                                    "image_id": next_img_id,
                                    "category_id": 1,
                                    "bbox": [0, 0, 0, 0],
                                    "keypoints": [0] * 51,
                                    "num_keypoints": 0,
                                    "iscrowd": 0,
                                }
                                next_img_id += 1
                                next_ann_id += 1

                            new_images.append(img_entry)
                            new_annotations.append(ann_entry)
                            self.img_ann_map[img_entry["id"]] = ann_entry
                            self.img_file_map[local_path] = img_entry

                self.coco_data["images"] = new_images
                self.coco_data["annotations"] = new_annotations
                self.status_bar.showMessage("Annotations successfully mapped.", 3000)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Load Error",
                    f"Failed to parse {os.path.basename(self.json_path)}: {e}. Starting fresh.",
                )
                self.initialize_fresh_coco()
        else:
            self.initialize_fresh_coco()

        # Check if loaded sequence matches the last saved session
        saved_settings = self.load_local_settings() or {}
        saved_dirs = saved_settings.get("camera_dirs")
        restore_frame_idx = 0
        if (
            isinstance(saved_dirs, dict)
            and len(saved_dirs) == len(CAMERA_KEYS)
            and all(
                os.path.abspath(saved_dirs.get(k, ""))
                == os.path.abspath(camera_dirs.get(k, ""))
                for k in CAMERA_KEYS
            )
        ):
            saved_frame_idx = saved_settings.get("current_frame_idx", 0)
            if 0 <= saved_frame_idx < len(self.sorted_frames):
                restore_frame_idx = saved_frame_idx

        if self.sorted_frames:
            self.slider_frame.setEnabled(True)
            self.slider_frame.setRange(0, len(self.sorted_frames) - 1)
            self.slider_frame.blockSignals(True)
            self.slider_frame.setValue(restore_frame_idx)
            self.slider_frame.blockSignals(False)

            self.spin_frame.setEnabled(True)
            self.spin_frame.setRange(1, len(self.sorted_frames))
            self.spin_frame.blockSignals(True)
            self.spin_frame.setValue(restore_frame_idx + 1)
            self.spin_frame.blockSignals(False)

            self.lbl_total_frames.setText(f"/ {len(self.sorted_frames)}")

        self.current_frame_idx = restore_frame_idx
        self.show_current_frame()
        self.global_3d_bounds = self.calculate_global_3d_bounds()

        self.btn_preprocess_seq.setEnabled(True)
        self.btn_clear_frame_ann.setEnabled(True)

        # Defer zoom to bounding boxes until layout finishes resizing at startup or sequence load
        QTimer.singleShot(0, self.zoom_all_bboxes)

        unannotated_count = 0
        for img in self.coco_data.get("images", []):
            ann = self.img_ann_map.get(img["id"])
            if ann and (not ann.get("bbox") or sum(ann["bbox"]) == 0):
                unannotated_count += 1

        if unannotated_count > 0:
            reply = QMessageBox.question(
                self,
                "Pre-processing Recommended",
                f"There are {unannotated_count} images without annotations in this sequence.\n"
                "Would you like to run automated pre-processing (YOLO + ViTPose) now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.run_sequence_preprocessing()
        log_debug("load_sequence_from_dirs completed successfully")

    def initialize_fresh_coco(self):
        """Initializes empty COCO dict structure mapping scanned local image files."""
        json_name = (
            os.path.basename(self.json_path) if self.json_path else "annotations.json"
        )
        self.status_bar.showMessage(f"Initializing new {json_name}...")
        self.coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "person"}],
        }
        self.img_ann_map.clear()
        self.img_file_map.clear()

        img_id = 1
        for frame_idx in self.sorted_frames:
            for cam_key in CAMERA_KEYS:
                if cam_key in self.frame_data[frame_idx]:
                    file_path = self.frame_data[frame_idx][cam_key]
                    img_entry = {
                        "id": img_id,
                        "file_name": file_path,
                        "width": 1920,
                        "height": 1080,
                    }
                    ann_entry = {
                        "id": img_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [0, 0, 0, 0],
                        "keypoints": [0] * 51,
                        "num_keypoints": 0,
                        "iscrowd": 0,
                    }
                    self.coco_data["images"].append(img_entry)
                    self.coco_data["annotations"].append(ann_entry)
                    self.img_ann_map[img_id] = ann_entry
                    self.img_file_map[file_path] = img_entry
                    img_id += 1
        self.status_bar.showMessage(f"New {json_name} initialized.", 3000)

    def show_current_frame(self, preserve_view=False):
        """Updates QGraphicsScene components on the 8 grid views."""
        log_debug(
            f"show_current_frame started, current_frame_idx={self.current_frame_idx}"
        )
        if self.current_frame_idx < 0 or self.current_frame_idx >= len(
            self.sorted_frames
        ):
            log_debug("show_current_frame early return due to bounds")
            return

        frame_idx = self.sorted_frames[self.current_frame_idx]
        log_debug(f"show_current_frame frame_idx={frame_idx}")
        self.frame_lbl.setText(
            f"Frame: {self.current_frame_idx + 1} / {len(self.sorted_frames)}"
        )
        self.status_bar.showMessage(f"Displaying frame index: {frame_idx}")

        maximized_id = self.get_maximized_camera_id()
        log_debug(f"show_current_frame maximized_id={maximized_id}")
        for i, key in enumerate(CAMERA_KEYS):
            cam_widget = self.camera_widgets[i]
            log_debug(f"show_current_frame processing camera={key} (i={i})")
            if key in self.frame_data[frame_idx]:
                img_path = self.frame_data[frame_idx][key]
                img_entry = self.img_file_map[img_path]
                ann = self.img_ann_map[img_entry["id"]]
                log_debug(
                    f"show_current_frame calling load_frame for key={key}, img={img_path}"
                )
                cam_widget.load_frame(img_path, ann, preserve_view=preserve_view)
                log_debug(f"show_current_frame load_frame done for key={key}")
                if maximized_id is None or i == maximized_id:
                    cam_widget.show()
                else:
                    cam_widget.hide()
            else:
                log_debug(f"show_current_frame key={key} missing from frame_data")
                cam_widget.scene.clear()
                txt_item = cam_widget.scene.addText(f"Missing frame data\nfor {key}")
                txt_item.setDefaultTextColor(QColor(148, 163, 184))
                if maximized_id is None or i == maximized_id:
                    cam_widget.show()
                else:
                    cam_widget.hide()

        # Update sidebar state
        log_debug("show_current_frame updating sidebar state")
        self.update_active_widgets_state()

        # Synchronize frame slider
        log_debug("show_current_frame syncing frame slider")
        self.slider_frame.blockSignals(True)
        self.slider_frame.setValue(self.current_frame_idx)
        self.slider_frame.blockSignals(False)

        # Synchronize frame spin box
        log_debug("show_current_frame syncing frame spin box")
        self.spin_frame.blockSignals(True)
        self.spin_frame.setValue(self.current_frame_idx + 1)
        self.spin_frame.blockSignals(False)

        # Update 3D skeleton visualization
        log_debug("show_current_frame updating 3D visualizer")
        self.update_3d_view()

        # Automatically persist settings (e.g. current_frame_idx)
        log_debug("show_current_frame saving local settings")
        self.save_local_settings()
        log_debug("show_current_frame completed successfully")

    def get_maximized_camera_id(self):
        """Returns the ID of the maximized view, or None."""
        for i, cam in enumerate(self.camera_widgets):
            if cam.is_maximized:
                return i
        return None

    def update_active_widgets_state(self):
        """Enables/disables buttons depending on maximized state."""
        maximized_id = self.get_maximized_camera_id()
        if maximized_id is not None:
            self.mode_lbl.setText(f"Maximized: {CAMERA_KEYS[maximized_id]}")
            self.btn_prev_view.show()
            self.btn_next_view.show()
            self.btn_prev_view.setEnabled(maximized_id > 0)
            self.btn_next_view.setEnabled(maximized_id < len(self.camera_widgets) - 1)
        else:
            self.mode_lbl.setText("Grid Mode (Double click view to zoom)")
            self.btn_prev_view.hide()
            self.btn_next_view.hide()

        # Enable zoom all if a sequence is loaded and has frames
        self.btn_zoom_all.setEnabled(
            self.sequence_dir is not None and len(self.sorted_frames) > 0
        )

        # Update ViTPose and Triangulation buttons state on all camera views
        worker_running = (
            self.active_worker is not None and self.active_worker.isRunning()
        )
        self.set_vitpose_buttons_enabled(not worker_running)
        self.set_triangulation_buttons_enabled(not worker_running)

    def set_vitpose_buttons_enabled(self, enabled):
        """Enables or disables the ViTPose button on all camera widgets."""
        for cam in self.camera_widgets:
            if hasattr(cam, "vitpose_btn") and cam.vitpose_btn:
                cam.vitpose_btn.setEnabled(enabled)

    def set_triangulation_buttons_enabled(self, enabled):
        """Enables or disables the Triangulate button on all camera widgets."""
        for cam in self.camera_widgets:
            if hasattr(cam, "triangulate_btn") and cam.triangulate_btn:
                cam.triangulate_btn.setEnabled(enabled)

    def toggle_maximize_camera(self, cam_id):
        """Maximizes double-clicked view to occupy full window space, or returns to grid."""
        cam = self.camera_widgets[cam_id]
        if not cam.is_maximized:
            # Hide all other camera widgets
            for i, c in enumerate(self.camera_widgets):
                if i != cam_id:
                    c.hide()
            # Stretch selected widget across all grid coordinates
            self.grid_layout.removeWidget(cam)
            self.grid_layout.addWidget(cam, 0, 0, 2, 4)
            cam.is_maximized = True

            # Auto-zoom to bounding box if it already exists (deferred for layout resize)
            QTimer.singleShot(0, cam.zoom_to_bbox)
        else:
            self.reset_camera_grid()

        self.update_active_widgets_state()

    def show_prev_camera_view(self):
        """Switches to the previous camera view when maximized."""
        maximized_id = self.get_maximized_camera_id()
        if maximized_id is not None and maximized_id > 0:
            self.switch_maximized_camera(maximized_id - 1)

    def show_next_camera_view(self):
        """Switches to the next camera view when maximized."""
        maximized_id = self.get_maximized_camera_id()
        if maximized_id is not None and maximized_id < len(self.camera_widgets) - 1:
            self.switch_maximized_camera(maximized_id + 1)

    def switch_maximized_camera(self, new_id):
        """Transitions the maximized state from the current view to a new view."""
        maximized_id = self.get_maximized_camera_id()
        if maximized_id is not None:
            old_cam = self.camera_widgets[maximized_id]
            self.grid_layout.removeWidget(old_cam)
            self.grid_layout.addWidget(
                old_cam, maximized_id // 4, maximized_id % 4, 1, 1
            )
            old_cam.is_maximized = False

        new_cam = self.camera_widgets[new_id]
        for i, c in enumerate(self.camera_widgets):
            if i != new_id:
                c.hide()
            else:
                c.show()

        self.grid_layout.removeWidget(new_cam)
        self.grid_layout.addWidget(new_cam, 0, 0, 2, 4)
        new_cam.is_maximized = True

        QTimer.singleShot(0, new_cam.zoom_to_bbox)
        self.update_active_widgets_state()

    def reset_camera_grid(self):
        """Resets the layout back to a 4x2 grid display."""
        maximized_id = self.get_maximized_camera_id()
        if maximized_id is not None:
            cam = self.camera_widgets[maximized_id]
            self.grid_layout.removeWidget(cam)
            # Explicitly add back with 1 row span and 1 column span
            self.grid_layout.addWidget(cam, maximized_id // 4, maximized_id % 4, 1, 1)
            cam.is_maximized = False

            # Show other widgets
            for c in self.camera_widgets:
                c.show()

            # Defer zoom to bounding boxes until layout finishes resizing
            QTimer.singleShot(0, self.zoom_all_bboxes)

            self.update_active_widgets_state()

    def zoom_active_bbox(self):
        """Zoom active view onto bounding box."""
        maximized_id = self.get_maximized_camera_id()
        if maximized_id is not None:
            self.camera_widgets[maximized_id].zoom_to_bbox()

    def zoom_all_bboxes(self):
        """Forces all 8 camera views to bbox mode and applies zoom/rotate to their bboxes."""
        for cam in self.camera_widgets:
            cam.zoom_to_bbox()

    def update_bbox(self, cam_id, bbox_coords, preserve_view=True):
        """Stores a bounding box into the memory model."""
        self.push_undo()
        frame_idx = self.sorted_frames[self.current_frame_idx]
        cam_key = CAMERA_KEYS[cam_id]
        img_path = self.frame_data[frame_idx][cam_key]
        img_id = self.img_file_map[img_path]["id"]

        ann = self.img_ann_map[img_id]
        ann["bbox"] = bbox_coords
        self.update_active_widgets_state()
        self.save_annotations()
        self.status_bar.showMessage(
            f"Updated bbox for camera {cam_id}: {bbox_coords}", 2000
        )

        # Refresh camera widget
        cam_widget = self.camera_widgets[cam_id]
        cam_widget.load_frame(img_path, ann, preserve_view=preserve_view)

    def update_keypoint(self, cam_id, point_id, x, y, save_and_sync=True):
        """Updates coordinates of keypoint and marks it as manually confirmed (v=2)."""
        frame_idx = self.sorted_frames[self.current_frame_idx]
        cam_key = CAMERA_KEYS[cam_id]
        img_path = self.frame_data[frame_idx][cam_key]
        img_id = self.img_file_map[img_path]["id"]

        ann = self.img_ann_map[img_id]
        offset = point_id * 3
        ann["keypoints"][offset] = float(x)
        ann["keypoints"][offset + 1] = float(y)
        ann["keypoints"][offset + 2] = 2  # Mark as manual adjustment

        # Calculate total annotated points
        ann["num_keypoints"] = sum(
            1 for idx in range(17) if ann["keypoints"][idx * 3 + 2] > 0
        )
        if save_and_sync:
            self.save_annotations()
            self.update_3d_view()

    def trigger_yolo_vitpose(self, camera_id=None):
        """Triggers the background thread to run ViTPose on the specified camera's bounding box."""
        if camera_id is False or camera_id is None:
            camera_id = self.get_maximized_camera_id()
        if camera_id is None:
            return

        if self.active_worker and self.active_worker.isRunning():
            self.status_bar.showMessage("A computation task is already in progress.")
            return

        frame_idx = self.sorted_frames[self.current_frame_idx]
        cam_key = CAMERA_KEYS[camera_id]
        img_path = self.frame_data[frame_idx][cam_key]
        img_entry = self.img_file_map[img_path]
        ann = self.img_ann_map[img_entry["id"]]
        bbox = ann.get("bbox", [0, 0, 0, 0])

        if not bbox or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            QMessageBox.warning(
                self,
                "No Bounding Box",
                f"Please draw a bounding box first (Shift + Drag) on camera {camera_id + 1}.",
            )
            return

        self.status_bar.showMessage(
            f"Running ViTPose on camera {camera_id} in background..."
        )
        self.set_triangulation_buttons_enabled(False)
        self.update_active_widgets_state()

        # Start background worker
        self.active_worker = WorkerThread(
            task_type="vitpose_only",
            model_wrapper=self.model_wrapper,
            args={
                "image_path": img_path,
                "camera_id": camera_id,
                "bbox": bbox,
                "threshold": getattr(self, "vitpose_threshold", 0.3),
            },
        )
        self.active_worker.finished.connect(self.on_yolo_vitpose_finished)
        self.active_worker.error.connect(self.on_worker_error)
        self.active_worker.start()

    def on_yolo_vitpose_finished(self, result):
        """Receives inference results from QThread and updates graphics scene."""
        cam_id = result["camera_id"]
        bbox = result["bbox"]
        keypoints = result["keypoints"]

        self.status_bar.showMessage(f"Inference completed for camera {cam_id}.", 3000)
        self.set_triangulation_buttons_enabled(True)
        self.update_active_widgets_state()

        # 1. Update bbox in model
        self.push_undo()
        frame_idx = self.sorted_frames[self.current_frame_idx]
        cam_key = CAMERA_KEYS[cam_id]
        img_path = self.frame_data[frame_idx][cam_key]
        img_entry = self.img_file_map[img_path]
        ann = self.img_ann_map[img_entry["id"]]
        ann["bbox"] = bbox

        # 2. Update keypoints in model
        if keypoints:
            flat_kps = []
            for kp in keypoints:
                flat_kps.extend(kp)
            ann["keypoints"] = flat_kps
            ann["num_keypoints"] = sum(
                1 for idx in range(17) if flat_kps[idx * 3 + 2] > 0
            )

        self.save_annotations()
        # 3. Update 3D triangulation and show new keypoints and reprojections across all 8 views
        self.update_3d_view()
        self.show_current_frame(preserve_view=False)
        self.zoom_all_bboxes()

    def on_worker_error(self, err_msg):
        self.status_bar.showMessage(f"Error: {err_msg}", 5000)
        QMessageBox.critical(
            self, "Model Error", f"An error occurred during inference:\n{err_msg}"
        )
        self.set_triangulation_buttons_enabled(True)
        self.update_active_widgets_state()

    def triangulate_view(self, cam_id):
        """Runs triangulation on other views and places/projects the resulting points on this camera view."""
        if self.active_worker and self.active_worker.isRunning():
            return

        self.push_undo()
        frame_idx = self.sorted_frames[self.current_frame_idx]

        # 1. Collect keypoints from all cameras
        keypoints_data = {}
        for c_id, key in enumerate(CAMERA_KEYS):
            if key in self.frame_data[frame_idx]:
                img_path = self.frame_data[frame_idx][key]
                img_id = self.img_file_map[img_path]["id"]
                flat_kps = self.img_ann_map[img_id]["keypoints"]
                kps = []
                for i in range(17):
                    kps.append(
                        [flat_kps[i * 3], flat_kps[i * 3 + 1], flat_kps[i * 3 + 2]]
                    )
                keypoints_data[c_id] = kps
            else:
                keypoints_data[c_id] = [[0.0, 0.0, 0]] * 17

        # 2. Build list of projection matrices
        matrices_list = []
        for key in CAMERA_KEYS:
            if key in self.camera_matrices:
                matrices_list.append(self.camera_matrices[key])
            else:
                matrices_list.append([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

        # 3. Perform triangulation and project onto target cam_id
        # We want to use all other cameras to perform the triangulation for cam_id
        updated_count = 0
        target_key = CAMERA_KEYS[cam_id]
        if target_key in self.frame_data[frame_idx]:
            img_path = self.frame_data[frame_idx][target_key]
            img_id = self.img_file_map[img_path]["id"]
            ann = self.img_ann_map[img_id]
            flat_kps = list(ann["keypoints"])

            for kp_idx in range(17):
                # Identify other cameras with valid annotations for this keypoint
                base_cams = []
                for c_id in range(8):
                    if c_id != cam_id:
                        kp = keypoints_data[c_id][kp_idx]
                        if kp[2] > 0:
                            base_cams.append(c_id)

                if len(base_cams) < 2:
                    continue

                # Build SVD matrix A from other views
                A = []
                for c_id in base_cams:
                    P = np.array(matrices_list[c_id])
                    u, v, _ = keypoints_data[c_id][kp_idx]

                    # Undistort
                    key = CAMERA_KEYS[c_id]
                    model_key = key.split("_")[1] if "_" in key else key
                    if self.calib_data and model_key in self.calib_data:
                        K = np.array(
                            self.calib_data[model_key]["matrix"], dtype=np.float32
                        )
                        distortions = np.array(
                            self.calib_data[model_key]["distortions"], dtype=np.float32
                        )
                        pt = np.array([[[u, v]]], dtype=np.float32)
                        undistorted_pt = cv2.undistortPoints(
                            pt, K, distortions, R=None, P=K
                        )
                        u, v = undistorted_pt[0, 0]

                    A.append(u * P[2, :] - P[0, :])
                    A.append(v * P[2, :] - P[1, :])

                valid = False
                A = np.array(A)
                _, _, Vt = np.linalg.svd(A)
                X = Vt[-1, :]
                if abs(X[3]) > 1e-5:
                    X = X / X[3]
                    X_3d = X[:3]
                    if np.all(np.abs(X_3d) < 50.0):
                        # Project back onto target camera cam_id
                        target_model_key = (
                            target_key.split("_")[1]
                            if "_" in target_key
                            else target_key
                        )
                        if self.calib_data and target_model_key in self.calib_data:
                            K = np.array(
                                self.calib_data[target_model_key]["matrix"],
                                dtype=np.float32,
                            )
                            distortions = np.array(
                                self.calib_data[target_model_key]["distortions"],
                                dtype=np.float32,
                            )
                            rvec = np.array(
                                self.calib_data[target_model_key]["rotation"],
                                dtype=np.float32,
                            )
                            tvec = np.array(
                                self.calib_data[target_model_key]["translation"],
                                dtype=np.float32,
                            )

                            img_pts, _ = cv2.projectPoints(
                                X_3d.reshape(1, 3), rvec, tvec, K, distortions
                            )
                            u_proj, v_proj = img_pts[0, 0]
                            valid = True
                        else:
                            P = np.array(matrices_list[cam_id])
                            X_homog = np.array([X_3d[0], X_3d[1], X_3d[2], 1.0])
                            x_proj = P @ X_homog
                            if x_proj[2] != 0:
                                u_proj = x_proj[0] / x_proj[2]
                                v_proj = x_proj[1] / x_proj[2]
                                valid = True
                            else:
                                valid = False

                if valid and 0.0 <= u_proj <= 1920.0 and 0.0 <= v_proj <= 1080.0:
                    flat_kps[kp_idx * 3] = float(u_proj)
                    flat_kps[kp_idx * 3 + 1] = float(v_proj)
                    flat_kps[kp_idx * 3 + 2] = (
                        2.0  # Labeled/confirmed via triangulation
                    )
                    updated_count += 1

            if updated_count > 0:
                ann["keypoints"] = flat_kps
                ann["num_keypoints"] = sum(
                    1 for idx in range(17) if flat_kps[idx * 3 + 2] > 0
                )
                self.camera_widgets[cam_id].load_frame(img_path, ann)
                self.save_annotations()
                self.global_3d_bounds = self.calculate_global_3d_bounds()
                self.update_3d_view()
                self.status_bar.showMessage(
                    f"Triangulated and placed {updated_count} points on view {cam_id}.",
                    3000,
                )
            else:
                self.status_bar.showMessage(
                    "Could not triangulate any points (need 2+ other views annotated).",
                    3000,
                )

    def run_sequence_preprocessing(self):
        """Spawns a progress dialog and starts background batch preprocessing of frames from the current frame to the end."""
        if (
            not self.coco_data
            or not self.coco_data.get("images")
            or not self.sorted_frames
        ):
            QMessageBox.warning(self, "No Sequence", "Please load a sequence first.")
            return

        if self.active_worker and self.active_worker.isRunning():
            QMessageBox.warning(
                self, "Active Task", "Another computation task is already in progress."
            )
            return

        current_frame_num = self.current_frame_idx + 1
        total_seq_frames = len(self.sorted_frames)

        reply = QMessageBox.question(
            self,
            "Confirm Pre-processing",
            f"Would you like to run preprocessing (YOLO + ViTPose) from the current frame "
            f"({current_frame_num}/{total_seq_frames}) to the end of the sequence?\n\n"
            "This will overwrite all existing bounding boxes and keypoints for these frames.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.No:
            return

        # Prepare frames subset to process
        frames_to_process = self.sorted_frames[self.current_frame_idx :]
        total_frames = len(frames_to_process)

        self.push_undo()

        # Clear existing annotations for the frames to process so they will be recalculated
        for frame_idx in frames_to_process:
            for cam_key in CAMERA_KEYS:
                path = self.frame_data[frame_idx].get(cam_key)
                if path:
                    img_entry = self.img_file_map.get(path)
                    if img_entry:
                        ann = self.img_ann_map.get(img_entry["id"])
                        if ann:
                            ann["bbox"] = [0, 0, 0, 0]
                            ann["keypoints"] = [0] * 51
                            ann["num_keypoints"] = 0

        # Create progress dialog
        self.progress_dialog = QProgressDialog(
            "Initializing models...", "Cancel", 0, total_frames, self
        )
        self.progress_dialog.setWindowTitle("Pre-processing Sequence")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)

        # Instantiate and start the worker thread
        self.preprocess_worker = SequencePreprocessWorker(
            model_wrapper=self.model_wrapper,
            sorted_frames=frames_to_process,
            frame_data=self.frame_data,
            img_file_map=self.img_file_map,
            img_ann_map=self.img_ann_map,
            threshold=getattr(self, "vitpose_threshold", 0.3),
        )
        self.active_worker = self.preprocess_worker

        # Connect signals
        self.preprocess_worker.progress.connect(self.on_preprocess_progress)
        self.preprocess_worker.finished.connect(self.on_preprocess_finished)
        self.preprocess_worker.error.connect(self.on_preprocess_error)

        # Handle cancel button clicked
        self.progress_dialog.canceled.connect(self.preprocess_worker.cancel)

        # Disable controls
        self.btn_preprocess_seq.setEnabled(False)
        self.btn_clear_frame_ann.setEnabled(False)
        self.set_triangulation_buttons_enabled(False)
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.update_active_widgets_state()

        self.preprocess_worker.start()

    def on_preprocess_progress(self, current, total, text):
        if hasattr(self, "progress_dialog") and self.progress_dialog:
            self.progress_dialog.setLabelText(text)
            self.progress_dialog.setValue(current)
        self.status_bar.showMessage(text)

    def on_preprocess_finished(self, count):
        self.active_worker = None
        self.status_bar.showMessage(
            f"Pre-processing completed. {count} images processed.", 5000
        )
        if hasattr(self, "progress_dialog") and self.progress_dialog:
            self.progress_dialog.close()

        # Re-enable controls
        self.btn_preprocess_seq.setEnabled(True)
        self.btn_clear_frame_ann.setEnabled(True)
        self.set_triangulation_buttons_enabled(True)
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)

        # Force all camera views to bbox mode
        for cam in self.camera_widgets:
            cam.zoom_to_bbox()

        self.update_active_widgets_state()

        # Save annotations automatically to JSON
        self.save_annotations()

        # Refresh UI
        self.show_current_frame()
        self.global_3d_bounds = self.calculate_global_3d_bounds()

        QMessageBox.information(
            self,
            "Pre-processing Completed",
            f"Pre-processing is complete.\n{count} images were successfully processed and saved.",
        )

    def on_preprocess_error(self, err_msg):
        self.active_worker = None
        self.status_bar.showMessage(f"Pre-processing error: {err_msg}", 5000)
        if hasattr(self, "progress_dialog") and self.progress_dialog:
            self.progress_dialog.close()

        self.btn_preprocess_seq.setEnabled(True)
        self.btn_clear_frame_ann.setEnabled(True)
        self.set_triangulation_buttons_enabled(True)
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)
        self.update_active_widgets_state()

        QMessageBox.critical(
            self, "Error", f"An error occurred during pre-processing:\n{err_msg}"
        )

    def prev_frame(self):
        """Navigate to previous frame."""
        if self.active_worker and self.active_worker.isRunning():
            return
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.show_current_frame()

    def next_frame(self):
        """Navigate to next frame."""
        if self.active_worker and self.active_worker.isRunning():
            return
        if self.current_frame_idx < len(self.sorted_frames) - 1:
            self.current_frame_idx += 1
            self.show_current_frame()

    def save_annotations(self):
        """Saves current annotations into the JSON file."""
        if self.active_worker and self.active_worker.isRunning():
            return
        if not self.json_path:
            return

        self.status_bar.showMessage(f"Saving to {os.path.basename(self.json_path)}...")
        try:
            with open(self.json_path, "w") as f:
                json.dump(self.coco_data, f, indent=2)
            self.status_bar.showMessage(f"Annotations saved to: {self.json_path}", 3000)
        except Exception as e:
            QMessageBox.critical(
                self, "Save Error", f"Could not save annotations file:\n{e}"
            )

    def update_keypoint_sizes(self, value):
        """Updates the visual size of keypoint markers in all graphics scenes."""
        self.keypoint_radius = value
        from src.items import ReprojectedPointItem

        for cam in self.camera_widgets:
            for kp in cam.keypoint_items.values():
                kp.set_radius(value)
            for item in cam.scene.items():
                if isinstance(item, ReprojectedPointItem):
                    item.set_radius(value)
        self.save_local_settings()

    def on_slider_frame_changed(self, value):
        """Called when the user drags the frame slider."""
        if self.current_frame_idx != value:
            self.current_frame_idx = value
            self.show_current_frame()

    def on_spin_frame_changed(self, value):
        """Called when the user types/changes the frame number in the spin box."""
        new_idx = value - 1
        if 0 <= new_idx < len(self.sorted_frames):
            if self.current_frame_idx != new_idx:
                self.current_frame_idx = new_idx
                self.show_current_frame()

    def show_settings(self):
        """Displays the settings dialog and updates views on change."""
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_val = dialog.chk_rotate.isChecked()
            if new_val != self.auto_rotate_enabled:
                self.auto_rotate_enabled = new_val
                # Refresh all views to apply or remove rotation
                for cam in self.camera_widgets:
                    if cam.view_mode == "bbox":
                        cam.apply_bbox_view()

    def toggle_3d_window(self):
        """Toggles visibility of the pop-out 3D visualizer window."""
        if self.visualizer_3d_window is None:
            self.visualizer_3d_window = Visualizer3DWindow(self)

        if self.visualizer_3d_window.isVisible():
            self.visualizer_3d_window.hide()
        else:
            total = len(self.sorted_frames) if self.sorted_frames else 0
            self.visualizer_3d_window.playback_slider.setRange(0, max(0, total - 1))
            self.visualizer_3d_window.playback_slider.setEnabled(total > 0)
            self.visualizer_3d_window.btn_play_pause.setEnabled(total > 0)
            self.visualizer_3d_window.btn_sync.setEnabled(total > 0)
            self.visualizer_3d_window.sync_to_annotator_frame()
            self.visualizer_3d_window.show()

    def update_3d_view(self):
        """Calculates 3D points and updates the inline plot and the 3D window if visible."""
        log_debug("update_3d_view started")
        pts_3d = self.calculate_3d_keypoints()
        log_debug("update_3d_view calculate_3d_keypoints done")
        if hasattr(self, "visualizer_3d_inline") and self.visualizer_3d_inline:
            log_debug("update_3d_view calling update_plot on inline visualizer")
            self.visualizer_3d_inline.update_plot(pts_3d)
            log_debug("update_3d_view update_plot on inline visualizer done")
        if self.visualizer_3d_window and self.visualizer_3d_window.isVisible():
            log_debug("update_3d_view 3D window is visible")
            if (
                self.visualizer_3d_window.playback_frame_idx != self.current_frame_idx
                or self.visualizer_3d_window.play_timer.isActive()
            ):
                log_debug("update_3d_view syncing 3D window to annotator frame")
                self.visualizer_3d_window.sync_to_annotator_frame()
            else:
                log_debug("update_3d_view updating 3D window visualization")
                self.visualizer_3d_window.update_visualization()
        log_debug("update_3d_view completed successfully")

    def calculate_3d_keypoints(self, frame_idx_in_list=None):
        """Calculates 3D coordinates for all 17 keypoints of the current or specified frame."""
        if frame_idx_in_list is None:
            frame_idx_in_list = self.current_frame_idx

        if (
            frame_idx_in_list < 0
            or not self.sorted_frames
            or frame_idx_in_list >= len(self.sorted_frames)
        ):
            return None

        frame_idx = self.sorted_frames[frame_idx_in_list]

        # Collect keypoints from all cameras
        keypoints_data = {}
        for cam_id, key in enumerate(CAMERA_KEYS):
            if key in self.frame_data[frame_idx]:
                img_path = self.frame_data[frame_idx][key]
                img_id = self.img_file_map[img_path]["id"]
                flat_kps = self.img_ann_map[img_id]["keypoints"]
                kps = []
                for i in range(17):
                    kps.append(
                        [flat_kps[i * 3], flat_kps[i * 3 + 1], flat_kps[i * 3 + 2]]
                    )
                keypoints_data[cam_id] = kps
            else:
                keypoints_data[cam_id] = [[0.0, 0.0, 0]] * 17

        # Get projection matrices
        matrices_list = []
        for key in CAMERA_KEYS:
            if key in self.camera_matrices:
                matrices_list.append(self.camera_matrices[key])
            else:
                matrices_list.append([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

        pts_3d = np.full((17, 3), np.nan)

        for kp_idx in range(17):
            annotated_cams = []
            for cam_id in range(8):
                kp = keypoints_data[cam_id][kp_idx]
                if kp[2] > 0:
                    annotated_cams.append(cam_id)

            if len(annotated_cams) < 2:
                continue

            A = []
            for cam_id in annotated_cams:
                P = np.array(matrices_list[cam_id])
                u, v, _ = keypoints_data[cam_id][kp_idx]

                # Undistort coordinates before DLT linear triangulation if calibration is available
                key = CAMERA_KEYS[cam_id]
                model_key = key.split("_")[1] if "_" in key else key
                if self.calib_data and model_key in self.calib_data:
                    K = np.array(self.calib_data[model_key]["matrix"], dtype=np.float32)
                    distortions = np.array(
                        self.calib_data[model_key]["distortions"], dtype=np.float32
                    )
                    pt = np.array([[[u, v]]], dtype=np.float32)
                    undistorted_pt = cv2.undistortPoints(
                        pt, K, distortions, R=None, P=K
                    )
                    u, v = undistorted_pt[0, 0]

                A.append(u * P[2, :] - P[0, :])
                A.append(v * P[2, :] - P[1, :])

            A = np.array(A)
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1, :]
            if abs(X[3]) > 1e-5:
                X = X / X[3]
                X_3d = X[:3]
                if np.all(np.abs(X_3d) < 50.0):
                    pts_3d[kp_idx] = X_3d

        return pts_3d

    def calculate_global_3d_bounds(self):
        """Calculates the global min and max coordinate bounds across all frames of the sequence."""
        if not self.sorted_frames:
            return None

        all_xs = []
        all_ys = []
        all_zs = []

        for idx in range(len(self.sorted_frames)):
            pts_3d = self.calculate_3d_keypoints(idx)
            if pts_3d is not None:
                valid_mask = ~np.isnan(pts_3d)
                valid_pts = pts_3d[
                    valid_mask[:, 0] & valid_mask[:, 1] & valid_mask[:, 2]
                ]
                if len(valid_pts) > 0:
                    all_xs.extend(valid_pts[:, 0])
                    all_ys.extend(valid_pts[:, 1])
                    all_zs.extend(valid_pts[:, 2])

        if not all_xs:
            return None

        return {
            "x_min": float(np.min(all_xs)),
            "x_max": float(np.max(all_xs)),
            "y_min": float(np.min(all_ys)),
            "y_max": float(np.max(all_ys)),
            "z_min": float(np.min(all_zs)),
            "z_max": float(np.max(all_zs)),
        }

    def save_local_settings(self):
        """Saves current settings and active frame to configs/local_settings.json."""
        try:
            settings = {
                "keypoint_radius": self.keypoint_radius,
                "auto_rotate_enabled": self.auto_rotate_enabled,
                "show_3d_reprojection": self.show_3d_reprojection,
                "realtime_triangulation_enabled": self.realtime_triangulation_enabled,
                "delete_bbox_on_clear": self.delete_bbox_on_clear,
                "vitpose_show_confidence": self.vitpose_show_confidence,
                "vitpose_threshold": self.vitpose_threshold,
                "camera_dirs": getattr(self, "camera_dirs", None),
                "current_frame_idx": self.current_frame_idx,
                "yolo_path": getattr(self, "yolo_path", None),
                "vitpose_path": getattr(self, "vitpose_path", None),
            }
            os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
            with open(SETTINGS_FILE, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def load_local_settings(self):
        """Loads settings from configs/local_settings.json if it exists."""
        if not os.path.exists(SETTINGS_FILE):
            return None
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
            return None

    def clear_current_frame_annotations(self):
        """Clears annotations (keypoints and optionally bboxes) for the current frame across all cameras."""
        if self.current_frame_idx < 0 or not self.sorted_frames:
            return

        # Push to undo stack
        self.push_undo()

        frame_idx = self.sorted_frames[self.current_frame_idx]
        log_debug(f"clear_current_frame_annotations started for frame_idx={frame_idx}")

        cleared_count = 0
        for cam_key in CAMERA_KEYS:
            if cam_key in self.frame_data[frame_idx]:
                img_path = self.frame_data[frame_idx][cam_key]
                img_entry = self.img_file_map.get(img_path)
                if img_entry:
                    ann = self.img_ann_map.get(img_entry["id"])
                    if ann:
                        # Clear keypoints
                        ann["keypoints"] = [0] * 51
                        ann["num_keypoints"] = 0

                        # Clear bbox if setting is active
                        if getattr(self, "delete_bbox_on_clear", False):
                            ann["bbox"] = [0, 0, 0, 0]
                        cleared_count += 1

        if cleared_count > 0:
            self.save_annotations()
            self.update_3d_view()
            self.show_current_frame(preserve_view=True)
            self.status_bar.showMessage(
                f"Cleared annotations on {cleared_count} camera views for the current frame.",
                3000,
            )
            log_debug(
                f"clear_current_frame_annotations finished: cleared {cleared_count} views"
            )

    def closeEvent(self, event):
        """Called when the window is closed. Save settings and state."""
        self.save_local_settings()
        event.accept()
