from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsRectItem
from PyQt6.QtGui import QColor, QPen, QBrush, QPainterPath, QPainterPathStroker
from PyQt6.QtCore import Qt, QPointF, QRectF, QTimer

from src.constants import KEYPOINT_COLORS

class KeypointItem(QGraphicsEllipseItem):
    """Interactive keypoint dot that updates positions in real-time when dragged."""
    def __init__(self, x, y, point_id, name, parent_widget, kv):
        radius = parent_widget.main_win.keypoint_radius if parent_widget and parent_widget.main_win else 6
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.setPos(x, y)
        self.point_id = point_id
        self.name = name
        self.parent_widget = parent_widget
        self.kv = kv
        
        # Color based on joint type
        color = KEYPOINT_COLORS.get(point_id, QColor(0, 255, 0))
        self.setBrush(QBrush(color))
        self.update_pen(radius)
        
        self.setFlags(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable | 
                      QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges |
                      QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsEllipseItem.GraphicsItemFlag.ItemIsFocusable)
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"{name} (ID: {point_id})")
        self.setZValue(5.0)
        
        # Group dragging states
        self._is_dragging_group = False
        self._selected_kps_to_drag = []
        self._selected_bboxes_to_drag = []
        self._drag_start_positions = {}
        self._drag_start_scene = QPointF()

    def update_pen(self, radius):
        """Scale border thickness dynamically with circle size."""
        pen_width = max(1.0, radius / 4.0)
        if self.kv == 1:
            self.setPen(QPen(QColor(234, 179, 8), pen_width, Qt.PenStyle.DashLine))
        else:
            self.setPen(QPen(Qt.GlobalColor.white, pen_width))

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionChange and self.parent_widget:
            new_pos = value
            # Notify the parent widget of the manual position adjustment (do not save/sync to disk/3D yet)
            self.parent_widget.update_keypoint_pos(self.point_id, new_pos.x(), new_pos.y(), save_and_sync=False)
            
            # Transition visibility to 2 (manual reference) immediately on drag
            if self.kv == 1:
                self.kv = 2
                radius = self.rect().width() / 2.0
                self.update_pen(radius)
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and getattr(self.parent_widget, 'delete_key_pressed', False):
            selected_items = self.parent_widget.scene.selectedItems()
            selected_kps = [item for item in selected_items if isinstance(item, KeypointItem)]
            if self in selected_kps:
                self.parent_widget.delete_multiple_keypoints(selected_kps)
                selected_bboxes = [item for item in selected_items if isinstance(item, BBoxItem)]
                if selected_bboxes:
                    for bbox in selected_bboxes:
                        bbox.delete_bbox()
            else:
                self.delete_point()
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            # Save history checkpoint before starting the drag
            if self.parent_widget and self.parent_widget.main_win:
                self.parent_widget.main_win.push_undo()
                
            # Multi-selection group drag initialization
            selected_items = self.parent_widget.scene.selectedItems()
            self._selected_kps_to_drag = [item for item in selected_items if isinstance(item, KeypointItem)]
            if self not in self._selected_kps_to_drag:
                self._selected_kps_to_drag.append(self)
                
            self._selected_bboxes_to_drag = [item for item in selected_items if isinstance(item, BBoxItem)]
            
            # Record starting position for each item in selection
            self._drag_start_positions = {}
            for kp in self._selected_kps_to_drag:
                self._drag_start_positions[kp] = kp.pos()
            for bbox in self._selected_bboxes_to_drag:
                self._drag_start_positions[bbox] = bbox.pos()
            
            self._drag_start_scene = event.scenePos()
            self._is_dragging_group = True
            self._has_moved = False
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if getattr(self, '_is_dragging_group', False):
            self._has_moved = True
            delta = event.scenePos() - self._drag_start_scene
            for kp in self._selected_kps_to_drag:
                start_pos = self._drag_start_positions[kp]
                kp.setPos(start_pos + delta)
            for bbox in getattr(self, '_selected_bboxes_to_drag', []):
                start_pos = self._drag_start_positions[bbox]
                bbox.setPos(start_pos + delta)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        has_moved = getattr(self, '_has_moved', False)
        if getattr(self, '_is_dragging_group', False):
            self._is_dragging_group = False
            
            # Save updated bboxes to memory database
            for bbox in getattr(self, '_selected_bboxes_to_drag', []):
                r = bbox.rect()
                p = bbox.pos()
                bbox_coords = [float(p.x() + r.x()), float(p.y() + r.y()), float(r.width()), float(r.height())]
                if bbox.parent_widget and bbox.parent_widget.current_annotation:
                    bbox.parent_widget.current_annotation["bbox"] = bbox_coords

            self._selected_kps_to_drag = []
            self._selected_bboxes_to_drag = []
            self._drag_start_positions = {}
            event.accept()
        super().mouseReleaseEvent(event)
        
        # Deselect if dragged (do this BEFORE reloading the frame to avoid C++ deletion issues)
        if has_moved and self.scene():
            self.scene().clearSelection()

        if self.parent_widget and self.parent_widget.main_win:
            main_win = self.parent_widget.main_win
            main_win.save_annotations()
            main_win.update_3d_view()
            main_win.show_current_frame(preserve_view=True)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_point()
            event.accept()
        else:
            super().keyPressEvent(event)

    def hoverEnterEvent(self, event):
        # Always display hovered point name in status bar, even when Delete key is held
        if self.parent_widget and self.parent_widget.main_win:
            self.parent_widget.main_win.status_bar.showMessage(f"Hovered Joint: {self.name} (ID: {self.point_id})")
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        if self.parent_widget and self.parent_widget.main_win:
            self.parent_widget.main_win.status_bar.clearMessage()
        super().hoverLeaveEvent(event)

    def delete_point(self):
        if self.parent_widget:
            self.parent_widget.delete_keypoint(self.point_id)

    def set_radius(self, radius):
        """Update keypoint circle size and border thickness."""
        self.setRect(-radius, -radius, radius * 2, radius * 2)
        self.update_pen(radius)


class SkeletonItem(QGraphicsLineItem):
    """Dynamic connection line between two KeypointItems."""
    def __init__(self, kp1, kp2, color):
        super().__init__()
        self.kp1 = kp1
        self.kp2 = kp2
        self.setPen(QPen(color, 2))  # Colored pen
        self.setZValue(3.0)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setEnabled(False)
        self.update_position()

    def update_position(self):
        if self.kp1 and self.kp2:
            self.setLine(self.kp1.pos().x(), self.kp1.pos().y(),
                         self.kp2.pos().x(), self.kp2.pos().y())


class BBoxItem(QGraphicsRectItem):
    """Draggable and resizable bounding box item."""
    def __init__(self, rect, parent_widget):
        super().__init__(rect)
        self.parent_widget = parent_widget
        
        # Style bounding box
        self.setPen(QPen(QColor(234, 179, 8), 2)) # Yellow
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        
        # Make item movable and selectable
        self.setFlags(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges |
                      QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable)
        
        self.setAcceptHoverEvents(True)
        self.handle_size = 8
        self.handles = {} # handle_position_name -> rect
        self.active_handle = None
        self.setZValue(1.0)
        self.update_handles()

        # Group dragging states
        self._selected_kps_to_drag = []
        self._selected_bboxes_to_drag = []
        self._drag_start_positions = {}

    def delete_bbox(self):
        # Save updated bounding box to memory database as empty [0,0,0,0]
        bbox_coords = [0.0, 0.0, 0.0, 0.0]
        # Defer execution to avoid deleting self within event handler
        QTimer.singleShot(0, lambda: self.parent_widget.main_win.update_bbox(self.parent_widget.camera_id, bbox_coords, preserve_view=False))

    def boundingRect(self):
        r = self.rect()
        s = self.handle_size
        hs = s / 2
        margin = 2 # Safety margin for pen thickness
        # Extend bounding box to include the resizing handles and pen borders
        return QRectF(r.left() - hs - margin, r.top() - hs - margin, r.width() + s + 2 * margin, r.height() + s + 2 * margin)

    def shape(self):
        """Define click hitbox shape to outline border and handles only, ignoring interior."""
        path = QPainterPath()
        
        # 1. outline borders with a thick stroke (16px click hitbox width)
        r = self.rect()
        border_path = QPainterPath()
        border_path.addRect(r)
        
        stroker = QPainterPathStroker()
        stroker.setWidth(6.0) # Click hitbox thickness
        stroked_border = stroker.createStroke(border_path)
        path.addPath(stroked_border)
        
        # 2. Add resize handles hitboxes
        for handle_rect in self.handles.values():
            path.addRect(handle_rect)
            
        return path

    def update_handles(self):
        self.prepareGeometryChange()
        r = self.rect()
        s = self.handle_size
        hs = s / 2
        
        # Top-left, Top-right, Bottom-left, Bottom-right corners
        self.handles = {
            "top_left": QRectF(r.left() - hs, r.top() - hs, s, s),
            "top_right": QRectF(r.right() - hs, r.top() - hs, s, s),
            "bottom_left": QRectF(r.left() - hs, r.bottom() - hs, s, s),
            "bottom_right": QRectF(r.right() - hs, r.bottom() - hs, s, s)
        }

    def paint(self, painter, option, widget):
        if self.isSelected():
            self.setPen(QPen(QColor(234, 179, 8), 2, Qt.PenStyle.DashLine))
        else:
            self.setPen(QPen(QColor(234, 179, 8), 2))
            
        super().paint(painter, option, widget)
        
        # Draw resize handles if selected or hovered
        painter.setPen(QPen(QColor(234, 179, 8), 1))
        painter.setBrush(QBrush(QColor(234, 179, 8)))
        for handle_rect in self.handles.values():
            painter.drawRect(handle_rect)

    def get_element_at_pos(self, pos):
        """Returns the corner handle name or edge name at the given position, or None."""
        # 1. Check corner handles first
        for name, handle_rect in self.handles.items():
            if handle_rect.contains(pos):
                return name
                
        # 2. Check edges if it's close to the border
        r = self.rect()
        px, py = pos.x(), pos.y()
        margin = 4.0 # Click margin thickness
        
        # Check top edge
        if abs(py - r.top()) <= margin and r.left() <= px <= r.right():
            return "top_edge"
        # Check bottom edge
        if abs(py - r.bottom()) <= margin and r.left() <= px <= r.right():
            return "bottom_edge"
        # Check left edge
        if abs(px - r.left()) <= margin and r.top() <= py <= r.bottom():
            return "left_edge"
        # Check right edge
        if abs(px - r.right()) <= margin and r.top() <= py <= r.bottom():
            return "right_edge"
            
        return None

    def update_cursor_shape(self, pos, modifiers):
        """Helper to set cursor shape without calling hoverMoveEvent with mouse event."""
        element = self.active_handle if self.active_handle else self.get_element_at_pos(pos)
        ctrl_pressed = modifiers & Qt.KeyboardModifier.ControlModifier
        
        if element is None:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif ctrl_pressed:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            angle = getattr(self.parent_widget, "current_rotation_angle", 0.0)
            angle = int(round(angle)) % 360
            
            if element in ("top_left", "bottom_right"):
                if angle in (90, 270):
                    self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                else:
                    self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif element in ("top_right", "bottom_left"):
                if angle in (90, 270):
                    self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                else:
                    self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif element in ("top_edge", "bottom_edge"):
                if angle in (90, 270):
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                else:
                    self.setCursor(Qt.CursorShape.SizeVerCursor)
            elif element in ("left_edge", "right_edge"):
                if angle in (90, 270):
                    self.setCursor(Qt.CursorShape.SizeVerCursor)
                else:
                    self.setCursor(Qt.CursorShape.SizeHorCursor)

    def hoverMoveEvent(self, event):
        pos = event.pos()
        element = self.get_element_at_pos(pos)
        self.active_handle = element
        self.update_cursor_shape(pos, event.modifiers())
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Delete only when the Delete key is held down and we left-click on the bbox
            if getattr(self.parent_widget, 'delete_key_pressed', False):
                selected_items = self.parent_widget.scene.selectedItems()
                selected_bboxes = [item for item in selected_items if isinstance(item, BBoxItem)]
                selected_kps = [item for item in selected_items if isinstance(item, KeypointItem)]
                if self in selected_bboxes:
                    for bbox in selected_bboxes:
                        bbox.delete_bbox()
                    if selected_kps:
                        self.parent_widget.delete_multiple_keypoints(selected_kps)
                else:
                    self.delete_bbox()
                event.accept()
                return
            pos = event.pos()
            element = self.get_element_at_pos(pos)
            self.active_handle = element
            
            ctrl_pressed = event.modifiers() & Qt.KeyboardModifier.ControlModifier
            
            if element is not None:
                if ctrl_pressed:
                    # Save history checkpoint before starting the drag
                    if self.parent_widget and self.parent_widget.main_win:
                        self.parent_widget.main_win.push_undo()

                    # Move mode
                    self._is_moving = True
                    self._is_resizing = False
                    self._drag_start_pos = event.scenePos()
                    self._has_moved = False
                    
                    # Group drag initialization for BBox move
                    selected_items = self.parent_widget.scene.selectedItems()
                    self._selected_kps_to_drag = [item for item in selected_items if isinstance(item, KeypointItem)]
                    self._selected_bboxes_to_drag = [item for item in selected_items if isinstance(item, BBoxItem)]
                    if self not in self._selected_bboxes_to_drag:
                        self._selected_bboxes_to_drag.append(self)
                        
                    self._drag_start_positions = {}
                    for kp in self._selected_kps_to_drag:
                        self._drag_start_positions[kp] = kp.pos()
                    for bbox in self._selected_bboxes_to_drag:
                        self._drag_start_positions[bbox] = bbox.pos()

                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                else:
                    # Resize mode
                    self._is_moving = False
                    self._is_resizing = True
                    self._resize_start_rect = self.rect()
                    self._resize_start_local = pos
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if getattr(self, '_is_moving', False):
            self._has_moved = True
            # Translate position in scene coordinates
            delta = event.scenePos() - self._drag_start_pos
            for kp in getattr(self, '_selected_kps_to_drag', []):
                start_pos = self._drag_start_positions[kp]
                kp.setPos(start_pos + delta)
            for bbox in getattr(self, '_selected_bboxes_to_drag', []):
                start_pos = self._drag_start_positions[bbox]
                bbox.setPos(start_pos + delta)
            event.accept()
        elif getattr(self, '_is_resizing', False) and self.active_handle:
            r = self._resize_start_rect
            local_pos = event.pos()
            dx = local_pos.x() - self._resize_start_local.x()
            dy = local_pos.y() - self._resize_start_local.y()
            
            min_size = 20.0
            
            new_left = r.left()
            new_top = r.top()
            new_right = r.right()
            new_bottom = r.bottom()
            
            # Corner dragging resizing
            if self.active_handle == "top_left":
                new_left = min(r.left() + dx, r.right() - min_size)
                new_top = min(r.top() + dy, r.bottom() - min_size)
            elif self.active_handle == "top_right":
                new_right = max(r.right() + dx, r.left() + min_size)
                new_top = min(r.top() + dy, r.bottom() - min_size)
            elif self.active_handle == "bottom_left":
                new_left = min(r.left() + dx, r.right() - min_size)
                new_bottom = max(r.bottom() + dy, r.top() + min_size)
            elif self.active_handle == "bottom_right":
                new_right = max(r.right() + dx, r.left() + min_size)
                new_bottom = max(r.bottom() + dy, r.top() + min_size)
            # Edge dragging resizing
            elif self.active_handle == "top_edge":
                new_top = min(r.top() + dy, r.bottom() - min_size)
            elif self.active_handle == "bottom_edge":
                new_bottom = max(r.bottom() + dy, r.top() + min_size)
            elif self.active_handle == "left_edge":
                new_left = min(r.left() + dx, r.right() - min_size)
            elif self.active_handle == "right_edge":
                new_right = max(r.right() + dx, r.left() + min_size)
                
            self.setRect(QRectF(new_left, new_top, new_right - new_left, new_bottom - new_top))
            self.update_handles()
            self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if getattr(self, '_is_moving', False):
            self._is_moving = False
            
            # Save updated bboxes to memory database
            for bbox in getattr(self, '_selected_bboxes_to_drag', []):
                r = bbox.rect()
                p = bbox.pos()
                bbox_coords = [float(p.x() + r.x()), float(p.y() + r.y()), float(r.width()), float(r.height())]
                if bbox.parent_widget and bbox.parent_widget.current_annotation:
                    bbox.parent_widget.current_annotation["bbox"] = bbox_coords
            
            # Clear variables
            self._selected_kps_to_drag = []
            self._selected_bboxes_to_drag = []
            self._drag_start_positions = {}
            self.active_handle = None
            
            # Update hover cursor shape cleanly
            self.update_cursor_shape(event.pos(), event.modifiers())
            
            # Clear selection if dragged (do this BEFORE reloading the frame/deleting items)
            if getattr(self, '_has_moved', False) and self.scene():
                self.scene().clearSelection()

            # Trigger save/3d updates
            if self.parent_widget and self.parent_widget.main_win:
                main_win = self.parent_widget.main_win
                main_win.save_annotations()
                main_win.update_3d_view()
                main_win.show_current_frame(preserve_view=True)
            
            event.accept()
        elif getattr(self, '_is_resizing', False):
            self._is_resizing = False
            self.active_handle = None
            
            # Save updated bounding box to memory database
            r = self.rect()
            p = self.pos()
            bbox_coords = [float(p.x() + r.x()), float(p.y() + r.y()), float(r.width()), float(r.height())]
            
            # Defer execution to avoid deleting self within event handler
            QTimer.singleShot(0, lambda: self.parent_widget.main_win.update_bbox(self.parent_widget.camera_id, bbox_coords))
            
            # Update hover cursor shape cleanly
            self.update_cursor_shape(event.pos(), event.modifiers())
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class ReprojectedPointItem(QGraphicsEllipseItem):
    """Hollow overlay circle showing the reprojected 3D coordinate for comparison."""
    def __init__(self, x, y, point_id, name, parent_widget):
        radius = parent_widget.main_win.keypoint_radius if parent_widget and parent_widget.main_win else 6
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.setPos(x, y)
        self.point_id = point_id
        self.name = name
        self.parent_widget = parent_widget
        
        # Distinct style: hollow, rose/magenta pen, dashed
        pen_width = max(1.0, radius / 4.0)
        self.setPen(QPen(QColor(244, 63, 94), pen_width, Qt.PenStyle.DashLine))
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        
        # Purely visual overlay, no interaction
        self.setFlags(QGraphicsEllipseItem.GraphicsItemFlag(0))
        self.setAcceptHoverEvents(False)
        self.setToolTip(f"Reprojected {name} (ID: {point_id})")
        self.setZValue(4.0)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setEnabled(False)

    def set_radius(self, radius):
        """Update reprojected circle size and border thickness."""
        self.setRect(-radius, -radius, radius * 2, radius * 2)
        pen_width = max(1.0, radius / 4.0)
        self.setPen(QPen(QColor(244, 63, 94), pen_width, Qt.PenStyle.DashLine))


class DiscrepancyLineItem(QGraphicsLineItem):
    """Dashed connection line between the user's manual keypoint and the 3D reprojected coordinate."""
    def __init__(self, x1, y1, x2, y2):
        super().__init__(x1, y1, x2, y2)
        self.setPen(QPen(QColor(244, 63, 94), 1, Qt.PenStyle.DotLine))
        self.setFlags(QGraphicsLineItem.GraphicsItemFlag(0))
        self.setZValue(2.0)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setEnabled(False)

