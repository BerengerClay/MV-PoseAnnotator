import os

from PyQt6.QtWidgets import (
    QDialog,
    QCheckBox,
    QDialogButtonBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QListView,
    QTreeView,
    QAbstractItemView,
    QSpinBox,
    QRadioButton,
    QButtonGroup,
)
from PyQt6.QtCore import Qt


def select_multiple_directories(
    parent=None, caption="Select Directories", directory=""
):
    """Opens a non-native file dialog allowing multiple directories to be selected."""
    dialog = QFileDialog(parent, caption, directory)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setFileMode(QFileDialog.FileMode.Directory)

    # Enable multiple/extended selection in the internal view widget
    for view in dialog.findChildren((QListView, QTreeView)):
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.selectedFiles()
    return []


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_win = parent
        self.setWindowTitle("Settings")
        self.resize(420, 550)

        if parent:
            self.original_kp_radius = parent.keypoint_radius
            self.original_rotate = parent.auto_rotate_enabled
            self.original_reproject = parent.show_3d_reprojection
            self.original_realtime_tri = getattr(
                parent, "realtime_triangulation_enabled", False
            )
            self.original_delete_bbox = getattr(
                parent, "delete_bbox_on_clear", False
            )
            self.original_show_confidence = getattr(
                parent, "vitpose_show_confidence", True
            )
            self.original_threshold = getattr(
                parent, "vitpose_threshold", 0.3
            )
            self.original_interpolated_opacity = getattr(
                parent, "interpolated_opacity", 0.4
            )
        else:
            self.original_kp_radius = 3
            self.original_rotate = True
            self.original_reproject = False
            self.original_realtime_tri = False
            self.original_delete_bbox = False
            self.original_show_confidence = True
            self.original_threshold = 0.3
            self.original_interpolated_opacity = 0.4

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Auto-rotation checkbox
        self.chk_rotate = QCheckBox(
            "Keep feet at bottom and head at top (auto-rotation)"
        )
        self.chk_rotate.setChecked(parent.auto_rotate_enabled if parent else True)
        if parent:
            self.chk_rotate.toggled.connect(self.on_rotate_toggled)
        layout.addWidget(self.chk_rotate)

        # Reprojection checkbox
        self.chk_reproject = QCheckBox("Show 3D reprojection overlay")
        self.chk_reproject.setChecked(parent.show_3d_reprojection if parent else False)
        if parent:
            self.chk_reproject.toggled.connect(self.on_reproject_toggled)
        layout.addWidget(self.chk_reproject)

        # Real-time triangulation checkbox
        self.chk_realtime_tri = QCheckBox(
            "Update 3D triangulation in real-time during drag"
        )
        self.chk_realtime_tri.setChecked(
            parent.realtime_triangulation_enabled if parent else False
        )
        if parent:
            self.chk_realtime_tri.toggled.connect(self.on_realtime_tri_toggled)
        layout.addWidget(self.chk_realtime_tri)

        # Delete bounding box on clear annotations checkbox
        self.chk_delete_bbox = QCheckBox(
            "Delete bounding boxes when clearing annotations"
        )
        self.chk_delete_bbox.setChecked(
            parent.delete_bbox_on_clear if parent else False
        )
        if parent:
            self.chk_delete_bbox.toggled.connect(self.on_delete_bbox_toggled)
        layout.addWidget(self.chk_delete_bbox)

        # Show confidence checkbox
        self.chk_show_confidence = QCheckBox("Show ViTPose confidence (opacity based on confidence)")
        self.chk_show_confidence.setChecked(parent.vitpose_show_confidence if parent else True)
        if parent:
            self.chk_show_confidence.toggled.connect(self.on_show_confidence_toggled)
        layout.addWidget(self.chk_show_confidence)

        # Keypoint size slider layout
        kp_size_layout = QHBoxLayout()
        kp_size_lbl = QLabel("Keypoint Size:")
        kp_size_lbl.setStyleSheet("color: #f8fafc; font-weight: bold; font-size: 12px;")

        self.slider_kp_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_kp_size.setRange(1, 10)
        self.slider_kp_size.setValue(self.original_kp_radius)
        if parent:
            self.slider_kp_size.valueChanged.connect(parent.update_keypoint_sizes)

        self.slider_kp_size.setStyleSheet("""
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
        kp_size_layout.addWidget(kp_size_lbl)
        kp_size_layout.addWidget(self.slider_kp_size)
        layout.addLayout(kp_size_layout)

        # ViTPose Threshold slider layout
        threshold_layout = QHBoxLayout()
        self.lbl_threshold = QLabel(f"ViTPose Threshold: {self.original_threshold:.2f}")
        self.lbl_threshold.setStyleSheet("color: #f8fafc; font-weight: bold; font-size: 12px;")

        self.slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_threshold.setRange(0, 100)
        self.slider_threshold.setValue(int(self.original_threshold * 100))
        self.slider_threshold.setStyleSheet("""
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
        self.slider_threshold.valueChanged.connect(self.on_threshold_changed)
        threshold_layout.addWidget(self.lbl_threshold)
        threshold_layout.addWidget(self.slider_threshold)
        layout.addLayout(threshold_layout)

        # Interpolated Opacity slider layout
        opacity_layout = QHBoxLayout()
        self.lbl_opacity = QLabel(f"Interpolated Opacity: {self.original_interpolated_opacity:.2f}")
        self.lbl_opacity.setStyleSheet("color: #f8fafc; font-weight: bold; font-size: 12px;")

        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(int(self.original_interpolated_opacity * 100))
        self.slider_opacity.setStyleSheet("""
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
        self.slider_opacity.valueChanged.connect(self.on_opacity_changed)
        opacity_layout.addWidget(self.lbl_opacity)
        opacity_layout.addWidget(self.slider_opacity)
        layout.addLayout(opacity_layout)

        # Help & Controls Box
        help_group = QGroupBox("Keyboard Shortcuts & Controls")
        help_group.setStyleSheet("""
            QGroupBox {
                color: #38bdf8;
                font-weight: bold;
                border: 1px solid #334155;
                margin-top: 10px;
                padding-top: 15px;
                border-radius: 6px;
                font-size: 12px;
            }
        """)
        help_layout = QVBoxLayout(help_group)

        help_text = (
            "<b>Mouse Interaction:</b><br>"
            "• Double-click view: Zoom in/out of the camera view<br>"
            "• Shift + Drag (Left Click): Draw bounding box<br>"
            "• BBox border corners: Drag to Resize bounding box<br>"
            "• BBox border lines: Drag to resize edge<br>"
            "• Ctrl + BBox border: Drag to Translate/Move bounding box<br>"
            "• Right-click + Drag: Pan zoomed camera canvas<br>"
            "• Mouse Wheel: Zoom in/out of camera canvas<br>"
            "• Click 3D Box icon on view overlay: Triangulate keypoints from other views onto this view<br><br>"
            "<b>Keyboard Actions:</b><br>"
            "• Delete / Backspace: Delete selected keypoint/BBox<br>"
            "• Delete + Left Click: Click-delete keypoint/BBox<br>"
            "• Insert: Open menu to add missing keypoints or bounding box<br><br>"
            "<b>Global Shortcuts:</b><br>"
            "  - <b>Left / Right Arrow</b>: Frame Navigation<br>"
            "  - <b>Escape</b>: Reset to 8-view Grid Mode<br>"
            "  - <b>Y</b>: Run ViTPose on maximized view's bounding box<br>"
            "  - <b>S</b>: Save sequence annotations to JSON file"
        )
        lbl_help = QLabel(help_text)
        lbl_help.setWordWrap(True)
        lbl_help.setStyleSheet("color: #94a3b8; font-size: 11px; line-height: 1.4;")
        help_layout.addWidget(lbl_help)
        layout.addWidget(help_group)

        layout.addSpacing(10)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #f8fafc;
            }
            QCheckBox {
                color: #f8fafc;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                background-color: #1e293b;
                border: 1px solid #475569;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background-color: #38bdf8;
                border-color: #0284c7;
            }
            QPushButton {
                background-color: #1e293b;
                color: white;
                border: 1px solid #334155;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #334155;
            }
        """)

    def on_rotate_toggled(self, checked):
        if self.parent_win:
            self.parent_win.auto_rotate_enabled = checked
            # Refresh camera orientations immediately
            for cam in self.parent_win.camera_widgets:
                if cam.view_mode == "bbox":
                    cam.apply_bbox_view()

    def on_reproject_toggled(self, checked):
        if self.parent_win:
            self.parent_win.show_3d_reprojection = checked
            self.parent_win.show_current_frame(preserve_view=True)

    def on_realtime_tri_toggled(self, checked):
        if self.parent_win:
            self.parent_win.realtime_triangulation_enabled = checked

    def on_delete_bbox_toggled(self, checked):
        if self.parent_win:
            self.parent_win.delete_bbox_on_clear = checked

    def on_show_confidence_toggled(self, checked):
        if self.parent_win:
            self.parent_win.vitpose_show_confidence = checked
            self.parent_win.show_current_frame(preserve_view=True)

    def on_threshold_changed(self, value):
        threshold_val = value / 100.0
        self.lbl_threshold.setText(f"ViTPose Threshold: {threshold_val:.2f}")
        if self.parent_win:
            self.parent_win.vitpose_threshold = threshold_val

    def on_opacity_changed(self, value):
        opacity_val = value / 100.0
        self.lbl_opacity.setText(f"Interpolated Opacity: {opacity_val:.2f}")
        if self.parent_win:
            self.parent_win.interpolated_opacity = opacity_val
            self.parent_win.show_current_frame(preserve_view=True)

    def accept(self):
        if self.parent_win:
            self.parent_win.save_local_settings()
        super().accept()

    def reject(self):
        if self.parent_win:
            self.parent_win.update_keypoint_sizes(self.original_kp_radius)
            self.parent_win.auto_rotate_enabled = self.original_rotate
            self.parent_win.show_3d_reprojection = self.original_reproject
            self.parent_win.realtime_triangulation_enabled = self.original_realtime_tri
            self.parent_win.delete_bbox_on_clear = self.original_delete_bbox
            self.parent_win.vitpose_show_confidence = self.original_show_confidence
            self.parent_win.vitpose_threshold = self.original_threshold
            self.parent_win.interpolated_opacity = self.original_interpolated_opacity
            # Re-apply orientations
            for cam in self.parent_win.camera_widgets:
                if cam.view_mode == "bbox":
                    cam.apply_bbox_view()
            self.parent_win.show_current_frame(preserve_view=True)
        super().reject()


class SelectCameraFoldersDialog(QDialog):
    def __init__(
        self, camera_keys, initial_parent="", prefilled_dirs=None, parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle("Select Camera Folders")
        self.resize(650, 450)
        self.camera_keys = camera_keys
        self.camera_dirs = {}
        self.initial_parent = initial_parent

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Style sheet
        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #f8fafc;
            }
            QLabel {
                color: #f8fafc;
                font-size: 12px;
            }
            QLineEdit {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QPushButton {
                background-color: #1e293b;
                color: white;
                border: 1px solid #334155;
                padding: 4px 10px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #334155;
            }
        """)

        main_layout.addWidget(QLabel("<b>Individual Camera Folders:</b>"))

        # 8 Camera folders rows
        self.cam_inputs = {}
        for key in camera_keys:
            row_layout = QHBoxLayout()
            cam_lbl = QLabel(f"{key}:")
            cam_lbl.setMinimumWidth(120)

            initial_val = prefilled_dirs.get(key, "") if prefilled_dirs else ""
            cam_txt = QLineEdit(initial_val)
            btn_cam_browse = QPushButton("Browse...")

            # Use default capture in lambda
            btn_cam_browse.clicked.connect(
                lambda checked=False, k=key: self.browse_camera(k)
            )

            row_layout.addWidget(cam_lbl)
            row_layout.addWidget(cam_txt, stretch=1)
            row_layout.addWidget(btn_cam_browse)
            main_layout.addLayout(row_layout)
            self.cam_inputs[key] = cam_txt

        main_layout.addStretch()

        # Dialog buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.validate_and_accept)
        self.buttons.rejected.connect(self.reject)
        main_layout.addWidget(self.buttons)

    def browse_camera(self, key):
        initial = self.cam_inputs[key].text()
        if not initial:
            initial = self.initial_parent
        dir_path = QFileDialog.getExistingDirectory(
            self, f"Select Folder for {key}", initial
        )
        if dir_path:
            self.cam_inputs[key].setText(dir_path)

    def validate_and_accept(self):
        # Retrieve and validate directories
        dirs = {}
        for key, input_widget in self.cam_inputs.items():
            path = input_widget.text().strip()
            if not path:
                QMessageBox.warning(
                    self, "Missing Folder", f"Please select a directory for {key}."
                )
                return
            if not os.path.isdir(path):
                QMessageBox.warning(
                    self,
                    "Invalid Folder",
                    f"The directory for {key} does not exist:\n{path}",
                )
                return
            # Check if directory is empty or has no images
            files = os.listdir(path)
            has_images = any(
                f.lower().endswith((".png", ".jpg", ".jpeg")) for f in files
            )
            if not has_images:
                QMessageBox.warning(
                    self,
                    "No Images",
                    f"The directory for {key} does not contain any images (.png, .jpg, .jpeg):\n{path}",
                )
                return
            dirs[key] = path

        self.camera_dirs = dirs
        self.accept()

    def get_camera_dirs(self):
        return self.camera_dirs


class PreprocessOptionsDialog(QDialog):
    def __init__(self, total_frames, current_frame_idx=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pre-processing Options")
        self.resize(400, 320)
        self.total_frames = total_frames
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Style sheet
        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #f8fafc;
            }
            QLabel {
                color: #f8fafc;
                font-size: 12px;
            }
            QSpinBox {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QCheckBox, QRadioButton {
                color: #f8fafc;
                font-size: 12px;
            }
            QCheckBox::indicator, QRadioButton::indicator {
                width: 16px;
                height: 16px;
                background-color: #1e293b;
                border: 1px solid #475569;
            }
            QCheckBox::indicator {
                border-radius: 4px;
            }
            QRadioButton::indicator {
                border-radius: 8px;
            }
            QCheckBox::indicator:checked, QRadioButton::indicator:checked {
                background-color: #38bdf8;
                border-color: #0284c7;
            }
            QPushButton {
                background-color: #1e293b;
                color: white;
                border: 1px solid #334155;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #334155;
            }
        """)

        layout.addWidget(QLabel("<b>Pre-processing Configuration:</b>"))

        # Radio buttons to run preprocessing
        self.rb_yolo_vitpose = QRadioButton("Run pre-processing (YOLO + ViTPose)")
        self.rb_yolo = QRadioButton("Run pre-processing (YOLO only)")
        self.rb_none = QRadioButton("No pre-processing (Step / Interpolation only)")
        self.rb_yolo_vitpose.setChecked(True)

        self.btn_group = QButtonGroup(self)
        self.btn_group.addButton(self.rb_yolo_vitpose)
        self.btn_group.addButton(self.rb_yolo)
        self.btn_group.addButton(self.rb_none)

        layout.addWidget(self.rb_yolo_vitpose)
        layout.addWidget(self.rb_yolo)
        layout.addWidget(self.rb_none)

        # Start frame
        start_layout = QHBoxLayout()
        start_lbl = QLabel("Starting frame index:")
        start_lbl.setMinimumWidth(180)
        self.spin_start = QSpinBox()
        self.spin_start.setRange(1, total_frames)
        self.spin_start.setValue(current_frame_idx + 1)
        start_layout.addWidget(start_lbl)
        start_layout.addWidget(self.spin_start)
        layout.addLayout(start_layout)

        # Frame step
        step_layout = QHBoxLayout()
        step_lbl = QLabel("Frame step:")
        step_lbl.setMinimumWidth(180)
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 100)
        self.spin_step.setValue(8)
        step_layout.addWidget(step_lbl)
        step_layout.addWidget(self.spin_step)
        layout.addLayout(step_layout)

        # Explanation box
        explanation = QLabel(
            "<i>Note: Intermediate frames (skipped by the step) will "
            "be automatically calculated by linear interpolation between the processed frames.</i>"
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(explanation)

        layout.addStretch()

        # Dialog buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def get_settings(self):
        mode = "none"
        if self.rb_yolo_vitpose.isChecked():
            mode = "yolo_vitpose"
        elif self.rb_yolo.isChecked():
            mode = "yolo"

        return {
            "run_preprocess": mode != "none",
            "preprocess_mode": mode,
            "start_frame_idx": self.spin_start.value() - 1,
            "frame_step": self.spin_step.value(),
        }

