from PyQt6.QtWidgets import QDialog, QCheckBox, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QGroupBox
from PyQt6.QtCore import Qt

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_win = parent
        self.setWindowTitle("Settings")
        self.resize(420, 520)
        
        if parent:
            self.original_kp_radius = parent.keypoint_radius
            self.original_rotate = parent.auto_rotate_enabled
            self.original_reproject = parent.show_3d_reprojection
            self.original_realtime_tri = getattr(parent, 'realtime_triangulation_enabled', False)
        else:
            self.original_kp_radius = 6
            self.original_rotate = True
            self.original_reproject = False
            self.original_realtime_tri = False
            
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Auto-rotation checkbox
        self.chk_rotate = QCheckBox("Keep feet at bottom and head at top (auto-rotation)")
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
        self.chk_realtime_tri = QCheckBox("Update 3D triangulation in real-time during drag")
        self.chk_realtime_tri.setChecked(parent.realtime_triangulation_enabled if parent else False)
        if parent:
            self.chk_realtime_tri.toggled.connect(self.on_realtime_tri_toggled)
        layout.addWidget(self.chk_realtime_tri)
        
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
            "• Mouse Wheel: Zoom in/out of camera canvas<br><br>"
            "<b>Keyboard Actions:</b><br>"
            "• Delete / Backspace: Delete selected keypoint/BBox<br>"
            "• Delete + Left Click: Click-delete keypoint/BBox<br>"
            "• Insert: Open menu to add missing keypoints or bounding box<br><br>"
            "<b>Global Shortcuts:</b><br>"
            "  - <b>Left / Right Arrow</b>: Frame Navigation<br>"
            "  - <b>Escape</b>: Reset to 8-view Grid Mode<br>"
            "  - <b>Y</b>: Run ViTPose on maximized view's bounding box<br>"
            "  - <b>T</b>: Run 3D Triangulation & projection<br>"
            "  - <b>S</b>: Save sequence annotations to JSON file"
        )
        lbl_help = QLabel(help_text)
        lbl_help.setWordWrap(True)
        lbl_help.setStyleSheet("color: #94a3b8; font-size: 11px; line-height: 1.4;")
        help_layout.addWidget(lbl_help)
        layout.addWidget(help_group)
        
        layout.addSpacing(10)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
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

    def reject(self):
        if self.parent_win:
            self.parent_win.update_keypoint_sizes(self.original_kp_radius)
            self.parent_win.auto_rotate_enabled = self.original_rotate
            self.parent_win.show_3d_reprojection = self.original_reproject
            self.parent_win.realtime_triangulation_enabled = self.original_realtime_tri
            # Re-apply orientations
            for cam in self.parent_win.camera_widgets:
                if cam.view_mode == "bbox":
                    cam.apply_bbox_view()
            self.parent_win.show_current_frame(preserve_view=True)
        super().reject()
