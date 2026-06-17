import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QPushButton,
    QLabel,
    QMessageBox,
    QMenu,
    QWidget,
    QHBoxLayout,
    QWidgetAction,
    QApplication,
)
from PyQt6.QtGui import QPixmap, QColor, QPen, QBrush, QCursor, QPainter
from PyQt6.QtCore import Qt, QPointF, QRectF, QSize

from src.constants import COCO_KEYPOINTS, COCO_SKELETON, CAMERA_KEYS, KEYPOINT_COLORS
from src.items import (
    KeypointItem,
    SkeletonItem,
    BBoxItem,
    ReprojectedPointItem,
    DiscrepancyLineItem,
)
from src.icons import get_lucide_icon, configure_button


def log_debug(msg):
    try:
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open("annotator.log", "a", encoding="utf-8") as f:
            f.write(f"[{now}] [Widgets] {msg}\n")
            f.flush()
    except Exception:
        pass


class CameraWidget(QGraphicsView):
    """Interactive graphics canvas for rendering a single camera view."""

    def __init__(self, camera_id, camera_name, main_win):
        super().__init__()
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.main_win = main_win

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.is_maximized = False

        self.pixmap_item = None
        self.bbox_item = None
        self.keypoint_items = {}
        self.skeleton_items = []

        # Panning states
        self._panning = False
        self._pan_start = QPointF()

        # Drawing bbox states
        self._drawing_bbox = False
        self._bbox_start_scene = QPointF()

        # View mode: "bbox" (zoomed & rotated) or "global" (unrotated, fit all)
        self.view_mode = "bbox"
        self.current_rotation_angle = 0.0
        self.manual_rotation_offset = 0.0
        self.current_annotation = None
        self.current_img_path = None
        self.user_has_zoomed_or_panned = False
        self.delete_key_pressed = False

        # Canvas settings
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(15, 23, 42)))  # Deep Slate Blue

        # Display label for camera name
        self.name_label = QLabel(camera_name, self)
        self.name_label.setStyleSheet(
            "color: #94a3b8; background-color: rgba(15, 23, 42, 180); padding: 4px 8px; border-radius: 4px; font-weight: bold;"
        )
        self.name_label.move(10, 10)

        # Toggle view button
        self.toggle_view_btn = QPushButton(self)
        self.toggle_view_btn.setIcon(get_lucide_icon("maximize-2", color="#f8fafc"))
        self.toggle_view_btn.setIconSize(QSize(12, 12))
        self.toggle_view_btn.setStyleSheet("""
            QPushButton {
                color: #f8fafc;
                background-color: rgba(30, 41, 59, 200);
                border: 1px solid #475569;
                padding: 2px 6px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: rgba(51, 65, 85, 220);
                border-color: #38bdf8;
            }
        """)
        self.toggle_view_btn.setToolTip("Switch between BBox Zoom and Global View")
        self.toggle_view_btn.clicked.connect(self.toggle_view_mode)
        self.toggle_view_btn.hide()  # Shown only when bbox is loaded

        # Clear annotations button
        self.delete_ann_btn = QPushButton(self)
        self.delete_ann_btn.setIcon(get_lucide_icon("trash-2", color="#f8fafc"))
        self.delete_ann_btn.setIconSize(QSize(12, 12))
        self.delete_ann_btn.setStyleSheet("""
            QPushButton {
                color: #f8fafc;
                background-color: rgba(239, 68, 68, 180);
                border: 1px solid #ef4444;
                padding: 2px 4px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: rgba(220, 38, 38, 220);
                border-color: #f87171;
            }
        """)
        self.delete_ann_btn.clicked.connect(self.clear_annotations)
        self.delete_ann_btn.setToolTip("Delete all keypoints on this view")
        self.delete_ann_btn.hide()

        # Swap Left/Right button
        self.swap_lr_btn = QPushButton(self)
        self.swap_lr_btn.setIcon(get_lucide_icon("arrow-right-left", color="#f8fafc"))
        self.swap_lr_btn.setIconSize(QSize(12, 12))
        self.swap_lr_btn.setStyleSheet("""
            QPushButton {
                color: #f8fafc;
                background-color: rgba(30, 41, 59, 200);
                border: 1px solid #475569;
                padding: 2px 4px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: rgba(51, 65, 85, 220);
                border-color: #38bdf8;
            }
        """)
        self.swap_lr_btn.clicked.connect(self.swap_left_right_keypoints)
        self.swap_lr_btn.setToolTip("Swap Left and Right keypoints for this view")
        self.swap_lr_btn.hide()

        # Copy previous frame annotations button
        self.copy_prev_btn = QPushButton(self)
        self.copy_prev_btn.setIcon(get_lucide_icon("history", color="#f8fafc"))
        self.copy_prev_btn.setIconSize(QSize(12, 12))
        self.copy_prev_btn.setStyleSheet("""
            QPushButton {
                color: #f8fafc;
                background-color: rgba(30, 41, 59, 200);
                border: 1px solid #475569;
                padding: 2px 4px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: rgba(51, 65, 85, 220);
                border-color: #38bdf8;
            }
        """)
        self.copy_prev_btn.clicked.connect(self.copy_keypoints_from_prev_frame)
        self.copy_prev_btn.setToolTip(
            "Copy annotations from previous frame for this view"
        )
        self.copy_prev_btn.hide()

        # Run ViTPose button on this view
        self.vitpose_btn = QPushButton(self)
        self.vitpose_btn.setIcon(get_lucide_icon("cpu", color="#f8fafc"))
        self.vitpose_btn.setIconSize(QSize(12, 12))
        self.vitpose_btn.setStyleSheet("""
            QPushButton {
                color: #f8fafc;
                background-color: rgba(30, 41, 59, 200);
                border: 1px solid #475569;
                padding: 2px 4px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: rgba(51, 65, 85, 220);
                border-color: #38bdf8;
            }
        """)
        self.vitpose_btn.clicked.connect(self.run_vitpose_on_this_view)
        self.vitpose_btn.setToolTip("Run ViTPose on this view's bounding box")
        self.vitpose_btn.hide()

        # Run Triangulation on this view
        self.triangulate_btn = QPushButton(self)
        self.triangulate_btn.setIcon(get_lucide_icon("box", color="#f8fafc"))
        self.triangulate_btn.setIconSize(QSize(12, 12))
        self.triangulate_btn.setStyleSheet("""
            QPushButton {
                color: #f8fafc;
                background-color: rgba(30, 41, 59, 200);
                border: 1px solid #475569;
                padding: 2px 4px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: rgba(51, 65, 85, 220);
                border-color: #38bdf8;
            }
        """)
        self.triangulate_btn.clicked.connect(self.run_triangulation_on_this_view)
        self.triangulate_btn.setToolTip("Triangulate points from other views and place on this view")
        self.triangulate_btn.hide()

        # Manual rotation buttons (bottom right)
        self.rotate_cw_btn = QPushButton(self)
        self.rotate_cw_btn.setIcon(get_lucide_icon("rotate-cw", color="#f8fafc"))
        self.rotate_cw_btn.setIconSize(QSize(12, 12))
        self.rotate_cw_btn.setStyleSheet("""
            QPushButton {
                color: #f8fafc;
                background-color: rgba(30, 41, 59, 200);
                border: 1px solid #475569;
                border-radius: 12px;
                padding: 4px;
            }
            QPushButton:hover {
                background-color: rgba(51, 65, 85, 220);
                border-color: #38bdf8;
            }
        """)
        self.rotate_cw_btn.clicked.connect(self.rotate_clockwise)
        self.rotate_cw_btn.setToolTip("Rotate view clockwise (90°)")
        self.rotate_cw_btn.hide()

        self.rotate_ccw_btn = QPushButton(self)
        self.rotate_ccw_btn.setIcon(get_lucide_icon("rotate-ccw", color="#f8fafc"))
        self.rotate_ccw_btn.setIconSize(QSize(12, 12))
        self.rotate_ccw_btn.setStyleSheet("""
            QPushButton {
                color: #f8fafc;
                background-color: rgba(30, 41, 59, 200);
                border: 1px solid #475569;
                border-radius: 12px;
                padding: 4px;
            }
            QPushButton:hover {
                background-color: rgba(51, 65, 85, 220);
                border-color: #38bdf8;
            }
        """)
        self.rotate_ccw_btn.clicked.connect(self.rotate_counter_clockwise)
        self.rotate_ccw_btn.setToolTip("Rotate view counter-clockwise (90°)")
        self.rotate_ccw_btn.hide()

    def start_panning(self, pos):
        """Start canvas panning with ClosedHandCursor override."""
        if not self._panning:
            self._panning = True
            self._pan_start = pos
            self._right_click_start = pos
            QApplication.setOverrideCursor(Qt.CursorShape.ClosedHandCursor)

    def stop_panning(self):
        """Stop canvas panning and restore cursor."""
        if self._panning:
            self._panning = False
            QApplication.restoreOverrideCursor()

    def hideEvent(self, event):
        self.stop_panning()
        super().hideEvent(event)

    def focusOutEvent(self, event):
        self.stop_panning()
        super().focusOutEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.main_win.toggle_maximize_camera(self.camera_id)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        """Interactive zoom centered on mouse cursor."""
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 0.85
        self.scale(zoom_factor, zoom_factor)
        self.user_has_zoomed_or_panned = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not getattr(self, "user_has_zoomed_or_panned", False):
            self.refresh_view()
        self.update_button_positions()

    def update_button_positions(self):
        """Update positions of overlay buttons based on which ones are currently visible."""
        w_toggle = 0
        w_delete = 0
        w_swap = 0
        w_copy = 0

        if hasattr(self, "toggle_view_btn") and self.toggle_view_btn:
            w_toggle = self.toggle_view_btn.sizeHint().width()
            if w_toggle <= 0:
                w_toggle = 65
            self.toggle_view_btn.resize(w_toggle, 20)
            self.toggle_view_btn.move(self.width() - w_toggle - 10, 10)

        if hasattr(self, "delete_ann_btn") and self.delete_ann_btn:
            w_delete = self.delete_ann_btn.sizeHint().width()
            if w_delete <= 0:
                w_delete = 28
            self.delete_ann_btn.resize(w_delete, 20)
            # Position it left of toggle_view_btn if visible, else top-right
            offset_x = (
                (w_toggle + 16)
                if (
                    hasattr(self, "toggle_view_btn")
                    and not self.toggle_view_btn.isHidden()
                )
                else 10
            )
            self.delete_ann_btn.move(self.width() - w_delete - offset_x, 10)

        if hasattr(self, "swap_lr_btn") and self.swap_lr_btn:
            w_swap = self.swap_lr_btn.sizeHint().width()
            if w_swap <= 0:
                w_swap = 28
            self.swap_lr_btn.resize(w_swap, 20)
            # Position it left of delete_ann_btn
            offset_toggle = (
                (w_toggle + 6)
                if (
                    hasattr(self, "toggle_view_btn")
                    and not self.toggle_view_btn.isHidden()
                )
                else 0
            )
            offset_delete = (
                (w_delete + 6)
                if (
                    hasattr(self, "delete_ann_btn")
                    and not self.delete_ann_btn.isHidden()
                )
                else 0
            )
            offset_x = offset_toggle + offset_delete + 10
            self.swap_lr_btn.move(self.width() - w_swap - offset_x, 10)

        if hasattr(self, "copy_prev_btn") and self.copy_prev_btn:
            w_copy = self.copy_prev_btn.sizeHint().width()
            if w_copy <= 0:
                w_copy = 28
            self.copy_prev_btn.resize(w_copy, 20)
            # Position it left of swap_lr_btn
            offset_toggle = (
                (w_toggle + 6)
                if (
                    hasattr(self, "toggle_view_btn")
                    and not self.toggle_view_btn.isHidden()
                )
                else 0
            )
            offset_delete = (
                (w_delete + 6)
                if (
                    hasattr(self, "delete_ann_btn")
                    and not self.delete_ann_btn.isHidden()
                )
                else 0
            )
            offset_swap = (
                (w_swap + 6)
                if (hasattr(self, "swap_lr_btn") and not self.swap_lr_btn.isHidden())
                else 0
            )
            offset_x = offset_toggle + offset_delete + offset_swap + 10
            self.copy_prev_btn.move(self.width() - w_copy - offset_x, 10)

        if hasattr(self, "vitpose_btn") and self.vitpose_btn:
            w_vit = self.vitpose_btn.sizeHint().width()
            if w_vit <= 0:
                w_vit = 28
            self.vitpose_btn.resize(w_vit, 20)
            # Position it left of copy_prev_btn
            offset_toggle = (
                (w_toggle + 6)
                if (
                    hasattr(self, "toggle_view_btn")
                    and not self.toggle_view_btn.isHidden()
                )
                else 0
            )
            offset_delete = (
                (w_delete + 6)
                if (
                    hasattr(self, "delete_ann_btn")
                    and not self.delete_ann_btn.isHidden()
                )
                else 0
            )
            offset_swap = (
                (w_swap + 6)
                if (hasattr(self, "swap_lr_btn") and not self.swap_lr_btn.isHidden())
                else 0
            )
            offset_copy = (
                (w_copy + 6)
                if (hasattr(self, "copy_prev_btn") and not self.copy_prev_btn.isHidden())
                else 0
            )
            offset_x = offset_toggle + offset_delete + offset_swap + offset_copy + 10
            self.vitpose_btn.move(self.width() - w_vit - offset_x, 10)

        if hasattr(self, "triangulate_btn") and self.triangulate_btn:
            w_tri = self.triangulate_btn.sizeHint().width()
            if w_tri <= 0:
                w_tri = 28
            self.triangulate_btn.resize(w_tri, 20)
            # Position it left of vitpose_btn
            offset_toggle = (
                (w_toggle + 6)
                if (
                    hasattr(self, "toggle_view_btn")
                    and not self.toggle_view_btn.isHidden()
                )
                else 0
            )
            offset_delete = (
                (w_delete + 6)
                if (
                    hasattr(self, "delete_ann_btn")
                    and not self.delete_ann_btn.isHidden()
                )
                else 0
            )
            offset_swap = (
                (w_swap + 6)
                if (hasattr(self, "swap_lr_btn") and not self.swap_lr_btn.isHidden())
                else 0
            )
            offset_copy = (
                (w_copy + 6)
                if (hasattr(self, "copy_prev_btn") and not self.copy_prev_btn.isHidden())
                else 0
            )
            offset_vit = (
                (w_vit + 6)
                if (hasattr(self, "vitpose_btn") and not self.vitpose_btn.isHidden())
                else 0
            )
            offset_x = offset_toggle + offset_delete + offset_swap + offset_copy + offset_vit + 10
            self.triangulate_btn.move(self.width() - w_tri - offset_x, 10)

        # Determine visibility of manual rotation buttons
        bbox = (
            self.current_annotation.get("bbox", [0, 0, 0, 0])
            if self.current_annotation
            else [0, 0, 0, 0]
        )
        has_bbox = bbox and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0
        show_rotate = self.view_mode == "bbox" and has_bbox

        if hasattr(self, "rotate_ccw_btn") and self.rotate_ccw_btn:
            if show_rotate:
                self.rotate_ccw_btn.show()
                self.rotate_ccw_btn.raise_()
            else:
                self.rotate_ccw_btn.hide()

        if hasattr(self, "rotate_cw_btn") and self.rotate_cw_btn:
            if show_rotate:
                self.rotate_cw_btn.show()
                self.rotate_cw_btn.raise_()
            else:
                self.rotate_cw_btn.hide()

        if show_rotate:
            margin_x = 10
            margin_y = 10
            btn_w, btn_h = 24, 24
            self.rotate_ccw_btn.resize(btn_w, btn_h)
            self.rotate_cw_btn.resize(btn_w, btn_h)

            rect = self.viewport().geometry()
            x_cw = rect.x() + rect.width() - btn_w - margin_x
            x_ccw = rect.x() + rect.width() - btn_w - btn_w - 6 - margin_x
            y_pos = rect.y() + rect.height() - btn_h - margin_y

            self.rotate_ccw_btn.move(x_ccw, y_pos)
            self.rotate_cw_btn.move(x_cw, y_pos)

    def mousePressEvent(self, event):
        self.setFocus()
        # Shift + Left click drag to draw manual bounding box
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() == Qt.KeyboardModifier.ShiftModifier
        ):
            self._drawing_bbox = True
            self._bbox_start_scene = self.mapToScene(event.pos())
            if self.bbox_item:
                self.scene.removeItem(self.bbox_item)
            self.bbox_item = self.scene.addRect(
                QRectF(self._bbox_start_scene, self._bbox_start_scene),
                QPen(QColor(234, 179, 8), 2, Qt.PenStyle.DashLine),
            )  # Yellow dash
            event.accept()
        # Right click to pan canvas or delete keypoint
        elif event.button() == Qt.MouseButton.RightButton:
            # Check if right-clicking on a KeypointItem
            clicked_item = None
            for item in self.items(event.pos()):
                if isinstance(item, KeypointItem):
                    clicked_item = item
                    break

            if clicked_item is not None:
                self.show_delete_keypoint_menu(clicked_item, event.globalPosition().toPoint())
                event.accept()
            else:
                self.start_panning(event.pos())
                event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drawing_bbox:
            curr_scene = self.mapToScene(event.pos())
            rect = QRectF(self._bbox_start_scene, curr_scene).normalized()
            self.bbox_item.setRect(rect)
            event.accept()
        elif self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            self.user_has_zoomed_or_panned = True
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drawing_bbox:
            self._drawing_bbox = False
            rect = self.bbox_item.rect()
            # Save the bounding box [x, y, w, h] to database
            bbox_coords = [rect.x(), rect.y(), rect.width(), rect.height()]
            self.main_win.update_bbox(self.camera_id, bbox_coords, preserve_view=False)
            self.bbox_item.setPen(QPen(QColor(234, 179, 8), 2))  # Solid yellow
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            click_start = getattr(self, "_right_click_start", None)
            drag_dist = 0
            if click_start is not None:
                drag_dist = (event.pos() - click_start).manhattanLength()

            self.stop_panning()
            event.accept()

            if click_start is not None and drag_dist < 5:
                self.show_insert_keypoint_menu()

            # Synthesize a mouse move event to update the graphics scene's hover state at the new position
            from PyQt6.QtGui import QMouseEvent
            from PyQt6.QtCore import QEvent
            pos = event.position()
            global_pos = event.globalPosition()
            fake_move = QMouseEvent(
                QEvent.Type.MouseMove,
                pos,
                global_pos,
                Qt.MouseButton.NoButton,
                Qt.MouseButton.NoButton,
                event.modifiers()
            )
            super().mouseMoveEvent(fake_move)
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self.delete_key_pressed = True

            # Delete multiple selected keypoints/bboxes if any are selected
            selected_items = self.scene.selectedItems()
            selected_kps = [
                item for item in selected_items if isinstance(item, KeypointItem)
            ]
            selected_bboxes = [
                item for item in selected_items if isinstance(item, BBoxItem)
            ]

            if selected_kps or selected_bboxes:
                if selected_kps:
                    self.delete_multiple_keypoints(selected_kps)
                if selected_bboxes:
                    for bbox in selected_bboxes:
                        bbox.delete_bbox()
                event.accept()
                return

        # Insert key to place a missing keypoint
        if event.key() == Qt.Key.Key_Insert:
            self.show_insert_keypoint_menu()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self.delete_key_pressed = False
        super().keyReleaseEvent(event)

    def delete_multiple_keypoints(self, kps):
        """Sets the visibility of multiple keypoints to 0 (hidden)."""
        self.main_win.push_undo()
        if hasattr(self, "current_annotation") and self.current_annotation:
            ann = self.current_annotation
            for kp in kps:
                offset = kp.point_id * 3
                ann["keypoints"][offset] = 0.0
                ann["keypoints"][offset + 1] = 0.0
                ann["keypoints"][offset + 2] = 0  # Visibility 0
            ann["num_keypoints"] = sum(
                1 for idx in range(17) if ann["keypoints"][idx * 3 + 2] > 0
            )

            # Reload to update visual skeleton and keypoints
            self.load_frame(self.current_img_path, ann, preserve_view=True)
            self.main_win.update_active_widgets_state()
            self.main_win.update_3d_view()
            self.main_win.save_annotations()
            if getattr(self.main_win, "show_3d_reprojection", False):
                self.main_win.show_current_frame(preserve_view=True)

    def load_frame(self, img_path, annotation, preserve_view=False):
        """Loads and draws image, bbox, keypoints, and skeleton."""
        self.stop_panning()
        log_debug(f"CameraWidget.load_frame started for camera {self.camera_name}, img_path={img_path}")
        # Save zoom/pan state if we are reloading the same image frame and preserve_view is requested
        is_same_image = (
            hasattr(self, "current_img_path")
            and self.current_img_path == img_path
            and self.scene.sceneRect().width() > 0
        )

        should_preserve = is_same_image and preserve_view

        if should_preserve:
            transform = self.transform()
            h_val = self.horizontalScrollBar().value()
            v_val = self.verticalScrollBar().value()

        # Preserve selected items state across scene clearing
        selected_point_ids = {kp.point_id for kp in self.keypoint_items.values() if kp.isSelected()}
        bbox_selected = self.bbox_item.isSelected() if (hasattr(self, 'bbox_item') and self.bbox_item is not None) else False

        log_debug(f"CameraWidget.load_frame clearing scene for camera {self.camera_name}")
        self.scene.clear()
        self.keypoint_items.clear()
        self.skeleton_items.clear()
        self.bbox_item = None

        self.current_img_path = img_path
        self.current_annotation = annotation

        if not os.path.exists(img_path):
            log_debug(f"CameraWidget.load_frame image not found for camera {self.camera_name}")
            txt_item = self.scene.addText(
                f"Image not found:\n{os.path.basename(img_path)}"
            )
            txt_item.setDefaultTextColor(QColor(239, 68, 68))  # Red text
            self.toggle_view_btn.hide()
            return

        # Load image
        pixmap = QPixmap(img_path)
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))

        # Draw bounding box and determine visibility of overlay buttons
        bbox = annotation.get("bbox", [0, 0, 0, 0])
        keypoints = annotation.get("keypoints", [])

        has_bbox = bbox and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0
        has_keypoints = keypoints and any(
            keypoints[idx * 3 + 2] > 0 for idx in range(17)
        )

        if has_bbox:
            x, y, w, h = bbox
            self.bbox_item = BBoxItem(QRectF(x, y, w, h), self)
            self.scene.addItem(self.bbox_item)
            if bbox_selected:
                self.bbox_item.setSelected(True)

            self.toggle_view_btn.show()
            self.toggle_view_btn.raise_()
            if self.view_mode == "bbox":
                configure_button(self.toggle_view_btn, icon_name="minimize-2")
            else:
                configure_button(self.toggle_view_btn, icon_name="maximize-2")

            # Show ViTPose button on this view since a bounding box exists
            self.vitpose_btn.show()
            self.vitpose_btn.raise_()
        else:
            self.bbox_item = None
            self.toggle_view_btn.hide()
            self.vitpose_btn.hide()

        if has_keypoints:
            self.delete_ann_btn.show()
            self.delete_ann_btn.raise_()
            self.swap_lr_btn.show()
            self.swap_lr_btn.raise_()
        else:
            self.delete_ann_btn.hide()
            self.swap_lr_btn.hide()

        # Show copy_prev_btn if we are not on the first frame of the sequence
        if self.main_win and self.main_win.current_frame_idx > 0:
            self.copy_prev_btn.show()
            self.copy_prev_btn.raise_()
        else:
            self.copy_prev_btn.hide()

        # Show triangulate_btn if sequence is loaded
        if self.main_win and self.main_win.sequence_dir is not None:
            self.triangulate_btn.show()
            self.triangulate_btn.raise_()
        else:
            self.triangulate_btn.hide()

        # Draw keypoints
        keypoints = annotation.get("keypoints", [])
        if keypoints:
            # keypoints is flat list: [x1, y1, v1, x2, y2, v2, ...]
            for idx in range(17):
                offset = idx * 3
                if offset + 2 < len(keypoints):
                    kx, ky, kv = (
                        keypoints[offset],
                        keypoints[offset + 1],
                        keypoints[offset + 2],
                    )
                    if kv > 0:
                        kp = KeypointItem(kx, ky, idx, COCO_KEYPOINTS[idx], self, kv)
                        self.scene.addItem(kp)
                        self.keypoint_items[idx] = kp
                        if idx in selected_point_ids:
                            kp.setSelected(True)

            # Draw skeleton lines
            for conn in COCO_SKELETON:
                p1, p2 = conn
                if p1 in self.keypoint_items and p2 in self.keypoint_items:
                    kp1 = self.keypoint_items[p1]
                    kp2 = self.keypoint_items[p2]

                    # Color segments by region (head, trunk, left, right)
                    if conn in [(5, 6), (11, 12)]:
                        # Torso/Trunk (Emerald Green)
                        color = QColor(16, 185, 129, 200)
                    elif conn in [(0, 1), (0, 2), (1, 3), (2, 4)]:
                        # Head/Face (Magenta/Pink)
                        color = QColor(236, 72, 153, 200)
                    elif p1 in [5, 7, 9, 11, 13, 15] and p2 in [5, 7, 9, 11, 13, 15]:
                        # Left side (Cyan)
                        color = QColor(6, 182, 212, 200)
                    else:
                        # Right side (Orange/Red)
                        color = QColor(249, 115, 22, 200)

                    line = SkeletonItem(kp1, kp2, color)
                    self.scene.addItem(line)
                    self.skeleton_items.append(line)

        # Draw reprojected 3D keypoints if enabled
        if self.main_win and getattr(self.main_win, "show_3d_reprojection", False):
            log_debug(f"CameraWidget.load_frame drawing 3D reprojections for camera {self.camera_name}")
            pts_3d = self.main_win.calculate_3d_keypoints()
            if pts_3d is not None and not np.all(np.isnan(pts_3d)):
                key = CAMERA_KEYS[self.camera_id]
                model_key = key.split("_")[1] if "_" in key else key
                calib_data = getattr(self.main_win, "calib_data", None)

                # Check if we have calibration data to do distorted projection
                use_distorted = calib_data and model_key in calib_data

                if use_distorted:
                    K = np.array(calib_data[model_key]["matrix"], dtype=np.float32)
                    D = np.array(calib_data[model_key]["distortions"], dtype=np.float32)
                    rvec = np.array(calib_data[model_key]["rotation"], dtype=np.float32)
                    tvec = np.array(
                        calib_data[model_key]["translation"], dtype=np.float32
                    )
                else:
                    P = self.main_win.camera_matrices.get(key)
                    if P is not None:
                        P = np.array(P)
                    else:
                        P = None

                for kp_idx in range(17):
                    X_3d = pts_3d[kp_idx]
                    if not np.isnan(X_3d[0]):
                        if use_distorted:
                            log_debug(f"CameraWidget.load_frame projecting distorted point {kp_idx} on camera {self.camera_name}")
                            img_pts, _ = cv2.projectPoints(
                                X_3d.reshape(1, 3), rvec, tvec, K, D
                            )
                            u_proj, v_proj = img_pts[0, 0]
                            valid = True
                        else:
                            if P is not None:
                                log_debug(f"CameraWidget.load_frame projecting undistorted point {kp_idx} on camera {self.camera_name}")
                                X_homog = np.array([X_3d[0], X_3d[1], X_3d[2], 1.0])
                                x_proj = P @ X_homog
                                if x_proj[2] != 0:
                                    u_proj = x_proj[0] / x_proj[2]
                                    v_proj = x_proj[1] / x_proj[2]
                                    valid = True
                                else:
                                    valid = False
                            else:
                                valid = False

                        if (
                            valid
                            and 0.0 <= u_proj <= 1920.0
                            and 0.0 <= v_proj <= 1080.0
                        ):
                            proj_item = ReprojectedPointItem(
                                u_proj, v_proj, kp_idx, COCO_KEYPOINTS[kp_idx], self
                            )
                            self.scene.addItem(proj_item)

                            # Draw a connection line if there is also an annotation
                            if kp_idx in self.keypoint_items:
                                kp_item = self.keypoint_items[kp_idx]
                                line_item = DiscrepancyLineItem(
                                    kp_item.pos().x(), kp_item.pos().y(), u_proj, v_proj
                                )
                                self.scene.addItem(line_item)

        # Refresh or restore view according to the current mode
        if should_preserve:
            self.setTransform(transform)
            self.horizontalScrollBar().setValue(h_val)
            self.verticalScrollBar().setValue(v_val)
        else:
            self.user_has_zoomed_or_panned = False
            self.refresh_view()

        # Update button positions dynamically
        self.update_button_positions()

    def get_body_up_vector(self, keypoints):
        """Finds 2D vector pointing from hips center to shoulders center."""
        # Find shoulders center
        shoulder_pts = []
        for idx in [5, 6]:
            offset = idx * 3
            if offset + 2 < len(keypoints) and keypoints[offset + 2] > 0:
                shoulder_pts.append((keypoints[offset], keypoints[offset + 1]))

        # Find hips center
        hip_pts = []
        for idx in [11, 12]:
            offset = idx * 3
            if offset + 2 < len(keypoints) and keypoints[offset + 2] > 0:
                hip_pts.append((keypoints[offset], keypoints[offset + 1]))

        if shoulder_pts and hip_pts:
            shoulder_x = sum(p[0] for p in shoulder_pts) / len(shoulder_pts)
            shoulder_y = sum(p[1] for p in shoulder_pts) / len(shoulder_pts)
            hip_x = sum(p[0] for p in hip_pts) / len(hip_pts)
            hip_y = sum(p[1] for p in hip_pts) / len(hip_pts)
            return shoulder_x - hip_x, shoulder_y - hip_y
        return None

    def apply_bbox_view(self):
        """Applies zoom and rotation to align body vertically."""
        old_anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self.resetTransform()
        self.horizontalScrollBar().setValue(0)
        self.verticalScrollBar().setValue(0)

        bbox = (
            self.current_annotation.get("bbox", [0, 0, 0, 0])
            if self.current_annotation
            else [0, 0, 0, 0]
        )
        keypoints = (
            self.current_annotation.get("keypoints", [])
            if self.current_annotation
            else []
        )

        angle = 0.0
        if getattr(self.main_win, "auto_rotate_enabled", True):
            if keypoints and sum(keypoints) > 0:
                vector = self.get_body_up_vector(keypoints)
                if vector:
                    vx, vy = vector
                    theta = np.arctan2(vy, vx)
                    theta_deg = np.degrees(theta)
                    raw_angle = -90.0 - theta_deg
                    # Snap to the nearest multiple of 90 degrees
                    angle = round(raw_angle / 90.0) * 90.0

        # Save rotation angle for cursor mapping in BBoxItem
        self.current_rotation_angle = (angle + self.manual_rotation_offset) % 360.0

        # Rotate view
        self.rotate(self.current_rotation_angle)

        # Fit in view with bounding box
        if bbox and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
            x, y, w, h = bbox
            # Add padding
            padding_x = w * 0.2
            padding_y = h * 0.2
            padded_rect = QRectF(
                x - padding_x, y - padding_y, w + 2 * padding_x, h + 2 * padding_y
            )
            self.fitInView(padded_rect, Qt.AspectRatioMode.KeepAspectRatio)
        else:
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        self.setTransformationAnchor(old_anchor)

    def apply_global_view(self):
        """Resets rotation and scales view to fit the entire scene."""
        old_anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self.resetTransform()
        self.horizontalScrollBar().setValue(0)
        self.verticalScrollBar().setValue(0)

        # Save rotation angle as 0.0
        self.current_rotation_angle = 0.0

        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        self.setTransformationAnchor(old_anchor)

    def refresh_view(self):
        """Applies either bbox/rotated view or global view depending on mode."""
        bbox = (
            self.current_annotation.get("bbox", [0, 0, 0, 0])
            if self.current_annotation
            else [0, 0, 0, 0]
        )
        if self.view_mode == "bbox" and bbox and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
            self.apply_bbox_view()
        else:
            self.apply_global_view()

    def toggle_view_mode(self):
        """Alternates between bounding box (zoomed/rotated) and global views."""
        if self.view_mode == "bbox":
            self.view_mode = "global"
            configure_button(self.toggle_view_btn, icon_name="maximize-2")
        else:
            self.view_mode = "bbox"
            configure_button(self.toggle_view_btn, icon_name="minimize-2")
        self.user_has_zoomed_or_panned = False
        self.refresh_view()
        self.update_button_positions()

    def zoom_to_bbox(self):
        """Forces bbox view mode and applies it."""
        self.view_mode = "bbox"
        configure_button(self.toggle_view_btn, icon_name="minimize-2")
        self.user_has_zoomed_or_panned = False
        self.apply_bbox_view()
        self.update_button_positions()

    def rotate_clockwise(self):
        """Manually rotate the view clockwise by 90 degrees in bbox zoom mode."""
        self.manual_rotation_offset = (self.manual_rotation_offset + 90.0) % 360.0
        self.user_has_zoomed_or_panned = False
        self.refresh_view()

    def rotate_counter_clockwise(self):
        """Manually rotate the view counter-clockwise by 90 degrees in bbox zoom mode."""
        self.manual_rotation_offset = (self.manual_rotation_offset - 90.0) % 360.0
        self.user_has_zoomed_or_panned = False
        self.refresh_view()

    def update_keypoint_pos(self, point_id, x, y, save_and_sync=False):
        """Triggered when user drags a KeypointItem. Updates local database and skeleton lines."""
        self.main_win.update_keypoint(
            self.camera_id, point_id, x, y, save_and_sync=save_and_sync
        )

        # Redraw skeleton lines connected to this joint
        for line in self.skeleton_items:
            if line.kp1.point_id == point_id or line.kp2.point_id == point_id:
                line.update_position()

        # If real-time 3D triangulation is enabled, refresh the 3D visualizer during drag.
        # Note: We do NOT call show_current_frame here because clearing the scene would delete
        # the KeypointItem object while it is still handling the drag event, causing a crash.
        # The 2D reprojected points will be updated when the user releases the mouse.
        if not save_and_sync and getattr(
            self.main_win, "realtime_triangulation_enabled", False
        ):
            self.main_win.update_3d_view()

    def delete_keypoint(self, point_id):
        """Sets the selected keypoint's visibility to 0 (hidden)."""
        self.main_win.push_undo()
        if hasattr(self, "current_annotation") and self.current_annotation:
            ann = self.current_annotation
            offset = point_id * 3
            ann["keypoints"][offset] = 0.0
            ann["keypoints"][offset + 1] = 0.0
            ann["keypoints"][offset + 2] = 0.0
            ann["num_keypoints"] = sum(
                1 for idx in range(17) if ann["keypoints"][idx * 3 + 2] > 0
            )

            # Reload to update visual skeleton and keypoints
            self.load_frame(self.current_img_path, ann, preserve_view=True)
            self.main_win.update_active_widgets_state()
            self.main_win.update_3d_view()
            self.main_win.save_annotations()
            if getattr(self.main_win, "show_3d_reprojection", False):
                self.main_win.show_current_frame(preserve_view=True)

    def clear_annotations(self):
        """Completely clears keypoints of this view after confirmation (retains bounding box)."""
        if not hasattr(self, "current_annotation") or not self.current_annotation:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Clear",
            f"Are you sure you want to delete all keypoints on {self.camera_name} (the bounding box will be preserved)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.main_win.push_undo()
            ann = self.current_annotation
            ann["keypoints"] = [0] * 51
            ann["num_keypoints"] = 0

            # Reload frame to refresh the canvas and hide buttons
            self.load_frame(self.current_img_path, ann, preserve_view=True)
            self.main_win.update_active_widgets_state()
            self.main_win.update_3d_view()
            self.main_win.save_annotations()
            if getattr(self.main_win, "show_3d_reprojection", False):
                self.main_win.show_current_frame(preserve_view=True)

    def show_insert_keypoint_menu(self):
        """Displays a context menu listing 'Ajouter Bounding Box' if none exists, or missing keypoints if one exists."""
        if not hasattr(self, "current_annotation") or not self.current_annotation:
            return

        ann = self.current_annotation
        bbox = ann.get("bbox", [0, 0, 0, 0])
        has_bbox = bbox and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0

        # Create context menu
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e293b;
                border: 1px solid #475569;
            }
            QMenu::item {
                padding: 6px 20px 6px 15px;
            }
            QMenu::item:selected {
                background-color: #334155;
            }
        """)

        actions = {}
        if not has_bbox:
            color = QColor(234, 179, 8)  # BBox Yellow
            action = QWidgetAction(menu)
            container = QWidget()
            container.setStyleSheet("background-color: transparent;")
            container.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

            layout = QHBoxLayout(container)
            layout.setContentsMargins(15, 6, 20, 6)
            layout.setSpacing(8)

            circle_lbl = QLabel()
            pixmap = QPixmap(10, 10)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(0, 0, 10, 10)
            painter.end()
            circle_lbl.setPixmap(pixmap)
            layout.addWidget(circle_lbl)

            name_lbl = QLabel("Add BBox")
            name_lbl.setStyleSheet("color: #f8fafc; font-weight: bold; font-size: 11px; background-color: transparent;")
            layout.addWidget(name_lbl)
            layout.addStretch()

            container.setLayout(layout)
            action.setDefaultWidget(container)
            menu.addAction(action)
            actions[action] = "add_bbox"
        else:
            keypoints = ann.get("keypoints", [])
            missing_kps = []
            for idx in range(17):
                offset = idx * 3
                if offset + 2 < len(keypoints) and keypoints[offset + 2] == 0:
                    missing_kps.append(idx)

            if not missing_kps:
                self.main_win.status_bar.showMessage(
                    "All keypoints are already present on this view.", 3000
                )
                return

            for idx in missing_kps:
                name = COCO_KEYPOINTS[idx]
                color = KEYPOINT_COLORS.get(idx, QColor(255, 255, 255))

                action = QWidgetAction(menu)
                container = QWidget()
                container.setStyleSheet("background-color: transparent;")
                container.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

                layout = QHBoxLayout(container)
                layout.setContentsMargins(15, 6, 20, 6)
                layout.setSpacing(8)

                # Color circle
                circle_lbl = QLabel()
                pixmap = QPixmap(10, 10)
                pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setBrush(QBrush(color))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(0, 0, 10, 10)
                painter.end()
                circle_lbl.setPixmap(pixmap)
                layout.addWidget(circle_lbl)

                # Text label styled in white
                name_lbl = QLabel(f"Add {name}")
                name_lbl.setStyleSheet("color: #f8fafc; font-weight: bold; font-size: 11px; background-color: transparent;")
                layout.addWidget(name_lbl)
                layout.addStretch()

                container.setLayout(layout)
                action.setDefaultWidget(container)
                menu.addAction(action)
                actions[action] = idx

        # Show menu at cursor pos
        global_pos = QCursor.pos()
        selected_action = menu.exec(global_pos)

        if selected_action in actions:
            val = actions[selected_action]
            if val == "add_bbox":
                scene_rect = self.scene.sceneRect()
                if scene_rect.width() > 0:
                    w = scene_rect.width() / 3
                    h = scene_rect.height() / 3
                    x = scene_rect.width() / 3
                    y = scene_rect.height() / 3
                else:
                    x, y, w, h = 640.0, 360.0, 640.0, 360.0
                bbox_coords = [float(x), float(y), float(w), float(h)]
                self.main_win.update_bbox(
                    self.camera_id, bbox_coords, preserve_view=False
                )
            else:
                idx = val
                # Convert screen coordinates to scene coordinates
                scene_pos = self.mapToScene(self.mapFromGlobal(global_pos))

                self.main_win.push_undo()
                # Place keypoint
                offset = idx * 3
                ann["keypoints"][offset] = float(scene_pos.x())
                ann["keypoints"][offset + 1] = float(scene_pos.y())
                ann["keypoints"][offset + 2] = (
                    2  # Visibility 2 (manual adjustment/confirmed)
                )
                ann["num_keypoints"] = sum(
                    1 for k in range(17) if ann["keypoints"][k * 3 + 2] > 0
                )

                # Reload frame to render the new keypoint
                self.load_frame(self.current_img_path, ann, preserve_view=True)
                self.main_win.update_active_widgets_state()
                self.main_win.save_annotations()

    def show_delete_keypoint_menu(self, keypoint_item, global_pos=None):
        """Displays a context menu to delete the specified keypoint with a styled layout."""
        if not hasattr(self, "current_annotation") or not self.current_annotation:
            return

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e293b;
                border: 1px solid #475569;
            }
            QMenu::item {
                padding: 6px 20px 6px 15px;
            }
            QMenu::item:selected {
                background-color: #334155;
            }
        """)

        # Get keypoint details
        idx = keypoint_item.point_id
        name = COCO_KEYPOINTS[idx]
        color = KEYPOINT_COLORS.get(idx, QColor(255, 255, 255))

        action = QWidgetAction(menu)
        container = QWidget()
        container.setStyleSheet("background-color: transparent;")
        container.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        layout = QHBoxLayout(container)
        layout.setContentsMargins(15, 6, 20, 6)
        layout.setSpacing(8)

        # Color circle
        circle_lbl = QLabel()
        pixmap = QPixmap(10, 10)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, 10, 10)
        painter.end()
        circle_lbl.setPixmap(pixmap)
        layout.addWidget(circle_lbl)

        # Text label styled in red or with standard text indicating deletion
        name_lbl = QLabel(f"Delete {name}")
        name_lbl.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 11px; background-color: transparent;")
        layout.addWidget(name_lbl)
        layout.addStretch()

        container.setLayout(layout)
        action.setDefaultWidget(container)
        menu.addAction(action)

        if global_pos is None:
            global_pos = QCursor.pos()

        selected_action = menu.exec(global_pos)
        if selected_action == action:
            self.delete_keypoint(idx)

    def run_vitpose_on_this_view(self):
        """Triggers ViTPose inference on this camera view's bounding box."""
        self.main_win.trigger_yolo_vitpose(self.camera_id)

    def run_triangulation_on_this_view(self):
        """Runs triangulation using other views and places points on this camera view."""
        self.main_win.triangulate_view(self.camera_id)

    def swap_left_right_keypoints(self):
        """Swaps left and right keypoints in the current annotation for this view."""
        if not hasattr(self, "current_annotation") or not self.current_annotation:
            return

        self.main_win.push_undo()
        ann = self.current_annotation
        keypoints = ann.get("keypoints", [])
        if not keypoints:
            return

        # Define COCO left/right pairs
        pairs = [
            (1, 2),  # eyes
            (3, 4),  # ears
            (5, 6),  # shoulders
            (7, 8),  # elbows
            (9, 10),  # wrists
            (11, 12),  # hips
            (13, 14),  # knees
            (15, 16),  # ankles
        ]

        for l_idx, r_idx in pairs:
            l_off = l_idx * 3
            r_off = r_idx * 3
            # Swap x, y, v
            keypoints[l_off], keypoints[r_off] = keypoints[r_off], keypoints[l_off]
            keypoints[l_off + 1], keypoints[r_off + 1] = (
                keypoints[r_off + 1],
                keypoints[l_off + 1],
            )
            keypoints[l_off + 2], keypoints[r_off + 2] = (
                keypoints[r_off + 2],
                keypoints[l_off + 2],
            )

        ann["keypoints"] = keypoints
        # Reload the frame preserving view scale/pan
        self.load_frame(self.current_img_path, ann, preserve_view=True)
        # Notify the main window to update 3D plot and active widgets states
        self.main_win.update_active_widgets_state()
        self.main_win.update_3d_view()
        self.main_win.save_annotations()
        if getattr(self.main_win, "show_3d_reprojection", False):
            self.main_win.show_current_frame(preserve_view=True)
        self.main_win.status_bar.showMessage(
            "Left/Right keypoints swapped for this view.", 3000
        )

    def copy_keypoints_from_prev_frame(self):
        """Copies keypoints and bounding box from the previous frame for the same camera view."""
        if not hasattr(self, "current_annotation") or not self.current_annotation:
            return

        main_win = self.main_win
        if main_win.current_frame_idx <= 0:
            main_win.status_bar.showMessage("Il n'y a pas de frame précédente.", 3000)
            return

        prev_frame_idx = main_win.sorted_frames[main_win.current_frame_idx - 1]
        key = self.camera_name

        if (
            prev_frame_idx not in main_win.frame_data
            or key not in main_win.frame_data[prev_frame_idx]
        ):
            main_win.status_bar.showMessage(
                f"Pas de données pour {key} à la frame précédente.", 3000
            )
            return

        prev_img_path = main_win.frame_data[prev_frame_idx][key]
        prev_img_entry = main_win.img_file_map.get(prev_img_path)
        if not prev_img_entry:
            return

        prev_ann = main_win.img_ann_map.get(prev_img_entry["id"])
        if not prev_ann:
            return

        prev_keypoints = prev_ann.get("keypoints", [])
        prev_bbox = prev_ann.get("bbox", [])

        has_prev_kps = prev_keypoints and any(
            prev_keypoints[idx * 3 + 2] > 0 for idx in range(17)
        )
        has_prev_bbox = prev_bbox and sum(prev_bbox) > 0

        if not has_prev_kps and not has_prev_bbox:
            main_win.status_bar.showMessage(
                f"Pas d'annotations à copier pour la frame précédente de {key}.", 3000
            )
            return

        main_win.push_undo()

        # Copy keypoints
        if has_prev_kps:
            self.current_annotation["keypoints"] = list(prev_keypoints)
            self.current_annotation["num_keypoints"] = prev_ann.get("num_keypoints", 0)
        else:
            self.current_annotation["keypoints"] = [0.0] * 51
            self.current_annotation["num_keypoints"] = 0

        # Copy bounding box
        if has_prev_bbox:
            self.current_annotation["bbox"] = list(prev_bbox)
        else:
            self.current_annotation["bbox"] = [0.0, 0.0, 0.0, 0.0]

        # Reload current frame
        self.load_frame(
            self.current_img_path, self.current_annotation, preserve_view=True
        )

        main_win.update_active_widgets_state()
        main_win.update_3d_view()
        main_win.save_annotations()
        if getattr(main_win, "show_3d_reprojection", False):
            main_win.show_current_frame(preserve_view=True)
        main_win.status_bar.showMessage(
            f"Annotations copiées depuis la frame précédente pour {key}.", 3000
        )
