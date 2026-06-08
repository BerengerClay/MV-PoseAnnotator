import sys
import os
import re
import json
import numpy as np
import torch

from PyQt6.QtWidgets import (QMainWindow, QWidget, QGridLayout, 
                             QVBoxLayout, QHBoxLayout, QPushButton, 
                             QFileDialog, QMessageBox, QLabel, QStatusBar, 
                             QProgressDialog, QSlider, QDialog, QStyle, QSpinBox)
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtCore import Qt, QTimer

from src.constants import CAMERA_KEYS
from src.widgets import CameraWidget
from src.workers import WorkerThread, SequencePreprocessWorker
from src.dialogs import SettingsDialog
from src.visualizer3d import Visualizer3DWindow, Visualizer3DWidget
from src.backend import ModelWrapper, triangulate_and_reproject
from src.icons import get_lucide_icon

class TrampolineAnnotator(QMainWindow):
    def __init__(self, sequence_dir=None):
        super().__init__()
        self.setWindowTitle("Multi-View Trampoline Jumper Annotator")
        self.setGeometry(100, 100, 1600, 950)
        
        # Application state
        self.sequence_dir = None
        self.json_path = None
        self.sorted_frames = []
        self.current_frame_idx = -1
        self.frame_data = {}  # frame_idx -> {camera_key -> filepath}
        
        self.coco_data = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "person"}]}
        self.img_ann_map = {}  # image_id -> annotation dict
        self.img_file_map = {} # file_name -> image dict
        
        # Load camera matrices
        self.camera_matrices = self.load_camera_matrices()
        
        # Deep learning models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_wrapper = ModelWrapper(weights_dir=None, device=self.device)
        self.active_worker = None
        self.keypoint_radius = 6 # Default radius
        self.visualizer_3d_window = None
        self.auto_rotate_enabled = True
        self.global_3d_bounds = None
        self.show_3d_reprojection = False
        self.realtime_triangulation_enabled = False

        # History stacks for Undo/Redo
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 50

        self.init_ui()
        self.apply_dark_style()
        self.setup_shortcuts()

        # Load sequence if provided via argument, otherwise prompt directory selection on startup
        if sequence_dir:
            self.load_sequence(sequence_dir)
        else:
            self.prompt_select_sequence()

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
        self.frame_lbl.setStyleSheet("color: #f8fafc; font-size: 18px; font-weight: bold;")
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
        self.btn_settings.setStyleSheet("background-color: #1e293b; border: 1px solid #334155;")
        
        # Maximize view indicators
        self.mode_lbl = QLabel("Grid Mode (Double click view to zoom)")
        self.mode_lbl.setStyleSheet("color: #38bdf8; font-weight: bold;")
        
        # AI commands and Triangulation
        ai_tri_layout = QHBoxLayout()
        
        self.btn_yolo_vit = QPushButton("Run ViTPose")
        self.btn_yolo_vit.setIcon(get_lucide_icon("cpu", color="#f8fafc"))
        self.btn_yolo_vit.clicked.connect(self.trigger_yolo_vitpose)
        self.btn_yolo_vit.setEnabled(False)
        self.btn_yolo_vit.setToolTip("Run ViTPose on the current active bounding box")
        
        self.btn_triangulate = QPushButton("Triangulate")
        self.btn_triangulate.setIcon(get_lucide_icon("box", color="#ffffff"))
        self.btn_triangulate.clicked.connect(self.trigger_triangulation)
        self.btn_triangulate.setStyleSheet("background-color: #059669; color: white;")
        self.btn_triangulate.setToolTip("Triangulate points labeled in 2+ cams and reproject on remaining views")
        
        ai_tri_layout.addWidget(self.btn_yolo_vit)
        ai_tri_layout.addWidget(self.btn_triangulate)
        
        self.btn_preprocess_seq = QPushButton("Preprocess Sequence")
        self.btn_preprocess_seq.setIcon(get_lucide_icon("sparkles", color="#ffffff"))
        self.btn_preprocess_seq.clicked.connect(self.run_sequence_preprocessing)
        self.btn_preprocess_seq.setEnabled(False)
        self.btn_preprocess_seq.setStyleSheet("background-color: #7c3aed; color: white; font-weight: bold;")
        self.btn_preprocess_seq.setToolTip("Run YOLO + ViTPose on all sequence images to generate preprocessing")

        self.btn_zoom_all = QPushButton("Zoom 8 Views to BBox")
        self.btn_zoom_all.setIcon(get_lucide_icon("maximize-2", color="#ffffff"))
        self.btn_zoom_all.clicked.connect(self.zoom_all_bboxes)
        self.btn_zoom_all.setEnabled(False)
        self.btn_zoom_all.setStyleSheet("background-color: #0369a1; color: white;")
        self.btn_zoom_all.setToolTip("Zoom and rotate all 8 camera views onto their bounding boxes")
        
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
        self.lbl_total_frames.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;")

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
        sidebar.addSpacing(15)
        sidebar.addWidget(self.btn_preprocess_seq)
        sidebar.addWidget(self.btn_zoom_all)
        sidebar.addSpacing(15)
        sidebar.addLayout(ai_tri_layout)
        
        sidebar.addStretch() # Push everything below to the bottom!
        
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

    def setup_shortcuts(self):
        """Registers global hotkeys to accelerate annotations."""
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self.prev_frame)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self.next_frame)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.reset_camera_grid)
        QShortcut(QKeySequence(Qt.Key.Key_Y), self, self.trigger_yolo_vitpose)
        QShortcut(QKeySequence(Qt.Key.Key_T), self, self.trigger_triangulation)
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
        if hasattr(self, 'btn_undo') and self.btn_undo:
            self.btn_undo.setEnabled(len(self.undo_stack) > 0)
        if hasattr(self, 'btn_redo') and self.btn_redo:
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
            QPushButton {
                background-color: #1e293b;
                border: 1px solid #334155;
                padding: 10px 15px;
                border-radius: 6px;
                font-weight: 500;
                font-size: 13px;
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
        """Open file dialog for the user to select the sequence folder."""
        initial_dir = "/usagers4/p123652/Documents/annotator"
        if not os.path.exists(initial_dir):
            initial_dir = os.path.expanduser("~")
            
        dir_path = QFileDialog.getExistingDirectory(self, "Select Sequence Directory", initial_dir)
        if dir_path:
            self.load_sequence(dir_path)

    def load_sequence(self, seq_dir):
        """Scans image files and loads or initializes annotations inside GT/."""
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.update_history_actions_state()
        self.sequence_dir = seq_dir
        self.path_lbl.setText(seq_dir)
        
        # Try Data folder, fallback to the sequence directory itself
        data_dir = os.path.join(seq_dir, "Data")
        if not os.path.exists(data_dir) or not any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(data_dir)):
            data_dir = seq_dir
            
        gt_dir = os.path.join(seq_dir, "GT")
        os.makedirs(gt_dir, exist_ok=True)
        
        # Find any json file in GT directory, default to annotations.json
        json_files = [f for f in os.listdir(gt_dir) if f.lower().endswith(".json")]
        if json_files:
            json_path = os.path.join(gt_dir, json_files[0])
        else:
            json_path = os.path.join(gt_dir, "annotations.json")
            
        self.json_path = json_path
        
        # 1. Scan the directory for camera frames
        self.status_bar.showMessage(f"Scanning directory: {os.path.basename(data_dir)}...")
        files = os.listdir(data_dir)
        self.frame_data.clear()
        
        for f in files:
            if not (f.lower().endswith(".png") or f.lower().endswith(".jpg") or f.lower().endswith(".jpeg")):
                continue
                
            # Frame index extraction: looks for frame_XXXXX in name
            match_frame = re.search(r"frame_(\d+)", f)
            if not match_frame:
                continue
            frame_idx = int(match_frame.group(1))
            
            # Camera key matching
            cam_key = None
            for key in CAMERA_KEYS:
                if key in f:
                    cam_key = key
                    break
                    
            if cam_key is None:
                continue
                
            if frame_idx not in self.frame_data:
                self.frame_data[frame_idx] = {}
            self.frame_data[frame_idx][cam_key] = os.path.join(data_dir, f)
            
        self.sorted_frames = sorted(list(self.frame_data.keys()))
        
        if not self.sorted_frames:
            QMessageBox.warning(self, "No frames found", "No valid camera frames found inside the 'Data' directory.")
            return

        # 2. Load or initialize annotations.json
        self.coco_data = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "person"}]}
        self.img_ann_map.clear()
        self.img_file_map.clear()
        
        if os.path.exists(json_path):
            self.status_bar.showMessage("Loading existing annotations.json...")
            try:
                with open(json_path, "r") as f:
                    self.coco_data = json.load(f)
                    
                # Re-index existing images and annotations by basename to support moving directories
                existing_ann = {}
                existing_img = {}
                for img in self.coco_data.get("images", []):
                    existing_img[os.path.basename(img["file_name"])] = img
                for ann in self.coco_data.get("annotations", []):
                    existing_ann[ann["image_id"]] = ann
                    
                # Build fresh maps matching our local files
                new_images = []
                new_annotations = []
                next_img_id = 1
                next_ann_id = 1
                
                # Check for highest IDs to avoid overlap
                if self.coco_data.get("images"):
                    next_img_id = max(img["id"] for img in self.coco_data["images"]) + 1
                if self.coco_data.get("annotations"):
                    next_ann_id = max(ann["id"] for ann in self.coco_data["annotations"]) + 1

                for frame_idx in self.sorted_frames:
                    for cam_key in CAMERA_KEYS:
                        if cam_key in self.frame_data[frame_idx]:
                            local_path = self.frame_data[frame_idx][cam_key]
                            base_name = os.path.basename(local_path)
                            
                            # Match with existing file entry
                            if base_name in existing_img:
                                img_entry = existing_img[base_name]
                                # Keep existing ID, update path to local file path
                                img_entry["file_name"] = local_path
                                ann_entry = existing_ann.get(img_entry["id"])
                                if ann_entry is None:
                                    # Create default annotation
                                    ann_entry = {
                                        "id": next_ann_id,
                                        "image_id": img_entry["id"],
                                        "category_id": 1,
                                        "bbox": [0, 0, 0, 0],
                                        "keypoints": [0] * 51,
                                        "num_keypoints": 0,
                                        "iscrowd": 0
                                    }
                                    next_ann_id += 1
                            else:
                                # Create new image entry
                                img_entry = {
                                    "id": next_img_id,
                                    "file_name": local_path,
                                    "width": 1920,
                                    "height": 1080
                                }
                                ann_entry = {
                                    "id": next_ann_id,
                                    "image_id": next_img_id,
                                    "category_id": 1,
                                    "bbox": [0, 0, 0, 0],
                                    "keypoints": [0] * 51,
                                    "num_keypoints": 0,
                                    "iscrowd": 0
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
                QMessageBox.critical(self, "Load Error", f"Failed to parse annotations.json: {e}. Starting fresh.")
                self.initialize_fresh_coco()
        else:
            self.initialize_fresh_coco()

        if self.sorted_frames:
            self.slider_frame.setEnabled(True)
            self.slider_frame.setRange(0, len(self.sorted_frames) - 1)
            self.slider_frame.setValue(0)
            self.spin_frame.setEnabled(True)
            self.spin_frame.setRange(1, len(self.sorted_frames))
            self.spin_frame.setValue(1)
            self.lbl_total_frames.setText(f"/ {len(self.sorted_frames)}")

        self.current_frame_idx = 0
        self.show_current_frame()
        self.global_3d_bounds = self.calculate_global_3d_bounds()
        
        self.btn_preprocess_seq.setEnabled(True)
        
        # Check if preprocessing is recommended
        unannotated_count = 0
        for img in self.coco_data.get("images", []):
            ann = self.img_ann_map.get(img["id"])
            if ann and (not ann.get("bbox") or sum(ann["bbox"]) == 0):
                unannotated_count += 1
                
        if unannotated_count > 0:
            reply = QMessageBox.question(
                self, "Pre-processing Recommended",
                f"There are {unannotated_count} images without annotations in this sequence.\n"
                "Would you like to run automated pre-processing (YOLO + ViTPose) now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.run_sequence_preprocessing()

    def initialize_fresh_coco(self):
        """Initializes empty COCO dict structure mapping scanned local image files."""
        self.status_bar.showMessage("Initializing new annotations.json...")
        self.coco_data = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "person"}]}
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
                        "height": 1080
                    }
                    ann_entry = {
                        "id": img_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [0, 0, 0, 0],
                        "keypoints": [0] * 51,
                        "num_keypoints": 0,
                        "iscrowd": 0
                    }
                    self.coco_data["images"].append(img_entry)
                    self.coco_data["annotations"].append(ann_entry)
                    self.img_ann_map[img_id] = ann_entry
                    self.img_file_map[file_path] = img_entry
                    img_id += 1
        self.status_bar.showMessage("New annotations.json initialized.", 3000)

    def show_current_frame(self, preserve_view=False):
        """Updates QGraphicsScene components on the 8 grid views."""
        if self.current_frame_idx < 0 or self.current_frame_idx >= len(self.sorted_frames):
            return
            
        frame_idx = self.sorted_frames[self.current_frame_idx]
        self.frame_lbl.setText(f"Frame: {self.current_frame_idx + 1} / {len(self.sorted_frames)}")
        self.status_bar.showMessage(f"Displaying frame index: {frame_idx}")
        
        maximized_id = self.get_maximized_camera_id()
        for i, key in enumerate(CAMERA_KEYS):
            cam_widget = self.camera_widgets[i]
            if key in self.frame_data[frame_idx]:
                img_path = self.frame_data[frame_idx][key]
                img_entry = self.img_file_map[img_path]
                ann = self.img_ann_map[img_entry["id"]]
                cam_widget.load_frame(img_path, ann, preserve_view=preserve_view)
                if maximized_id is None or i == maximized_id:
                    cam_widget.show()
                else:
                    cam_widget.hide()
            else:
                cam_widget.scene.clear()
                cam_widget.scene.addText(f"Missing frame data\nfor {key}", QPen(QColor(148, 163, 184)).color())
                if maximized_id is None or i == maximized_id:
                    cam_widget.show()
                else:
                    cam_widget.hide()

        # Update sidebar state
        self.update_active_widgets_state()

        # Synchronize frame slider
        self.slider_frame.blockSignals(True)
        self.slider_frame.setValue(self.current_frame_idx)
        self.slider_frame.blockSignals(False)

        # Synchronize frame spin box
        self.spin_frame.blockSignals(True)
        self.spin_frame.setValue(self.current_frame_idx + 1)
        self.spin_frame.blockSignals(False)

        # Update 3D skeleton visualization
        self.update_3d_view()

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
            self.btn_yolo_vit.setEnabled(True)
            # Check if active view has a bbox
            cam = self.camera_widgets[maximized_id]
            # self.btn_zoom.setEnabled(cam.bbox_item is not None)
        else:
            self.mode_lbl.setText("Grid Mode (Double click view to zoom)")
            self.btn_yolo_vit.setEnabled(False)
            # self.btn_zoom.setEnabled(False)

        # Enable zoom all if a sequence is loaded and has frames
        self.btn_zoom_all.setEnabled(self.sequence_dir is not None and len(self.sorted_frames) > 0)

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
        self.status_bar.showMessage(f"Updated bbox for camera {cam_id}: {bbox_coords}", 2000)

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
        ann["keypoints"][offset+1] = float(y)
        ann["keypoints"][offset+2] = 2  # Mark as manual adjustment
        
        # Calculate total annotated points
        ann["num_keypoints"] = sum(1 for idx in range(17) if ann["keypoints"][idx*3 + 2] > 0)
        if save_and_sync:
            self.save_annotations()
            self.update_3d_view()

    def trigger_yolo_vitpose(self):
        """Triggers the background thread to run ViTPose on the active view's bounding box."""
        maximized_id = self.get_maximized_camera_id()
        if maximized_id is None:
            return
            
        if self.active_worker and self.active_worker.isRunning():
            self.status_bar.showMessage("A computation task is already in progress.")
            return

        frame_idx = self.sorted_frames[self.current_frame_idx]
        cam_key = CAMERA_KEYS[maximized_id]
        img_path = self.frame_data[frame_idx][cam_key]
        img_entry = self.img_file_map[img_path]
        ann = self.img_ann_map[img_entry["id"]]
        bbox = ann.get("bbox", [0, 0, 0, 0])

        if not bbox or sum(bbox) == 0:
            QMessageBox.warning(self, "No Bounding Box", 
                                "Please draw a bounding box first (Shift + Drag) on this view.")
            return
        
        self.status_bar.showMessage(f"Running ViTPose on camera {maximized_id} in background...")
        self.btn_yolo_vit.setEnabled(False)
        self.btn_triangulate.setEnabled(False)
        
        # Start background worker
        self.active_worker = WorkerThread(
            task_type="vitpose_only",
            model_wrapper=self.model_wrapper,
            args={"image_path": img_path, "camera_id": maximized_id, "bbox": bbox}
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
        self.btn_yolo_vit.setEnabled(True)
        self.btn_triangulate.setEnabled(True)
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
            ann["num_keypoints"] = sum(1 for idx in range(17) if flat_kps[idx*3 + 2] > 0)
        
        self.save_annotations()
        # 3. Reload camera view preserving current viewport scaling/panning
        self.camera_widgets[cam_id].load_frame(img_path, ann, preserve_view=True)

    def on_worker_error(self, err_msg):
        self.status_bar.showMessage(f"Error: {err_msg}", 5000)
        QMessageBox.critical(self, "Model Error", f"An error occurred during inference:\n{err_msg}")
        self.btn_yolo_vit.setEnabled(True)
        self.btn_triangulate.setEnabled(True)
        self.update_active_widgets_state()

    def trigger_triangulation(self):
        """Runs Direct Linear Transformation 3D Triangulation and projects results."""
        if self.active_worker and self.active_worker.isRunning():
            return
        self.push_undo()
        frame_idx = self.sorted_frames[self.current_frame_idx]
        
        # Collect keypoints from all cameras of the current frame
        keypoints_data = {}
        for cam_id, key in enumerate(CAMERA_KEYS):
            if key in self.frame_data[frame_idx]:
                img_path = self.frame_data[frame_idx][key]
                img_id = self.img_file_map[img_path]["id"]
                flat_kps = self.img_ann_map[img_id]["keypoints"]
                
                # Reshape keypoints flat list to lists of [x, y, v]
                kps = []
                for i in range(17):
                    kps.append([flat_kps[i*3], flat_kps[i*3 + 1], flat_kps[i*3 + 2]])
                keypoints_data[cam_id] = kps
            else:
                keypoints_data[cam_id] = [[0.0, 0.0, 0]] * 17

        # Convert projection matrices dict to lists matching CAMERA_KEYS order
        matrices_list = []
        for key in CAMERA_KEYS:
            if key in self.camera_matrices:
                matrices_list.append(self.camera_matrices[key])
            else:
                # Identity matrix fallback (should not occur if config is correct)
                matrices_list.append([[1,0,0,0], [0,1,0,0], [0,0,1,0]])

        # Execute triangulation math
        try:
            updated_kps = triangulate_and_reproject(keypoints_data, matrices_list)
            
            # Save predictions back to memory database
            for cam_id, kps in updated_kps.items():
                key = CAMERA_KEYS[cam_id]
                if key in self.frame_data[frame_idx]:
                    img_path = self.frame_data[frame_idx][key]
                    img_id = self.img_file_map[img_path]["id"]
                    ann = self.img_ann_map[img_id]
                    
                    flat_kps = []
                    for kp in kps:
                        flat_kps.extend(kp)
                    ann["keypoints"] = flat_kps
                    ann["num_keypoints"] = sum(1 for idx in range(17) if flat_kps[idx*3 + 2] > 0)
                    
                    # Refresh widget display
                    self.camera_widgets[cam_id].load_frame(img_path, ann)
                    
            self.status_bar.showMessage("Triangulation and reprojection completed successfully.", 4000)
            self.save_annotations()
            self.global_3d_bounds = self.calculate_global_3d_bounds()
            self.update_3d_view()
        except Exception as e:
            QMessageBox.critical(self, "Triangulation Error", f"Could not perform triangulation:\n{e}")

    def run_sequence_preprocessing(self):
        """Spawns a progress dialog and starts background batch preprocessing of all frames."""
        if not self.coco_data or not self.coco_data.get("images"):
            QMessageBox.warning(self, "No Sequence", "Please load a sequence first.")
            return

        if self.active_worker and self.active_worker.isRunning():
            QMessageBox.warning(self, "Active Task", "Another computation task is already in progress.")
            return

        total_images = len(self.coco_data["images"])
        
        # Count how many actually need preprocessing
        to_process = 0
        for img in self.coco_data["images"]:
            ann = self.img_ann_map[img["id"]]
            if not ann.get("bbox") or sum(ann["bbox"]) == 0:
                to_process += 1
                
        if to_process == 0:
            reply = QMessageBox.question(
                self, "Already Processed",
                "All images already have annotations.\n"
                "Would you like to recalculate (overwrite) everything?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
            else:
                self.push_undo()
                for img in self.coco_data["images"]:
                    ann = self.img_ann_map[img["id"]]
                    ann["bbox"] = [0, 0, 0, 0]
                    ann["keypoints"] = [0] * 51
                    ann["num_keypoints"] = 0
                to_process = total_images
        else:
            self.push_undo()

        # Create progress dialog
        self.progress_dialog = QProgressDialog("Initializing models...", "Cancel", 0, total_images, self)
        self.progress_dialog.setWindowTitle("Pre-processing Sequence")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        
        # Instantiate and start the worker thread
        self.preprocess_worker = SequencePreprocessWorker(
            model_wrapper=self.model_wrapper,
            images=self.coco_data["images"],
            img_ann_map=self.img_ann_map
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
        self.btn_yolo_vit.setEnabled(False)
        self.btn_triangulate.setEnabled(False)
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        
        self.preprocess_worker.start()

    def on_preprocess_progress(self, current, total, text):
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.setLabelText(text)
            self.progress_dialog.setValue(current)
        self.status_bar.showMessage(text)

    def on_preprocess_finished(self, count):
        self.active_worker = None
        self.status_bar.showMessage(f"Pre-processing completed. {count} images processed.", 5000)
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            
        # Re-enable controls
        self.btn_preprocess_seq.setEnabled(True)
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
            self, "Pre-processing Completed",
            f"Pre-processing is complete.\n{count} images were successfully processed and saved."
        )

    def on_preprocess_error(self, err_msg):
        self.active_worker = None
        self.status_bar.showMessage(f"Pre-processing error: {err_msg}", 5000)
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            
        self.btn_preprocess_seq.setEnabled(True)
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)
        self.update_active_widgets_state()
        
        QMessageBox.critical(
            self, "Error",
            f"An error occurred during pre-processing:\n{err_msg}"
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
            QMessageBox.critical(self, "Save Error", f"Could not save annotations file:\n{e}")

    def update_keypoint_sizes(self, value):
        """Updates the visual size of keypoint markers in all graphics scenes."""
        self.keypoint_radius = value
        for cam in self.camera_widgets:
            for kp in cam.keypoint_items.values():
                kp.set_radius(value)

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
        pts_3d = self.calculate_3d_keypoints()
        if hasattr(self, 'visualizer_3d_inline') and self.visualizer_3d_inline:
            self.visualizer_3d_inline.update_plot(pts_3d)
        if self.visualizer_3d_window and self.visualizer_3d_window.isVisible():
            if self.visualizer_3d_window.playback_frame_idx != self.current_frame_idx or self.visualizer_3d_window.play_timer.isActive():
                self.visualizer_3d_window.sync_to_annotator_frame()
            else:
                self.visualizer_3d_window.update_visualization()

    def calculate_3d_keypoints(self, frame_idx_in_list=None):
        """Calculates 3D coordinates for all 17 keypoints of the current or specified frame."""
        if frame_idx_in_list is None:
            frame_idx_in_list = self.current_frame_idx
            
        if frame_idx_in_list < 0 or not self.sorted_frames or frame_idx_in_list >= len(self.sorted_frames):
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
                    kps.append([flat_kps[i*3], flat_kps[i*3 + 1], flat_kps[i*3 + 2]])
                keypoints_data[cam_id] = kps
            else:
                keypoints_data[cam_id] = [[0.0, 0.0, 0]] * 17
                
        # Get projection matrices
        matrices_list = []
        for key in CAMERA_KEYS:
            if key in self.camera_matrices:
                matrices_list.append(self.camera_matrices[key])
            else:
                matrices_list.append([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
                
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
                A.append(u * P[2, :] - P[0, :])
                A.append(v * P[2, :] - P[1, :])
                
            A = np.array(A)
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1, :]
            if X[3] != 0:
                X = X / X[3]
                pts_3d[kp_idx] = X[:3]
                
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
                valid_pts = pts_3d[valid_mask[:, 0] & valid_mask[:, 1] & valid_mask[:, 2]]
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
            "z_max": float(np.max(all_zs))
        }

