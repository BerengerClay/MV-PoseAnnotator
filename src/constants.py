from PyQt6.QtGui import QColor

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),               # Face
    (5, 6),                                       # Shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),              # Arms
    (5, 11), (6, 12), (11, 12),                   # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)        # Legs
]

KEYPOINT_COLORS = {
    # Face
    0: QColor(255, 0, 255), 1: QColor(0, 205, 100), 2: QColor(255, 255, 100), 3: QColor(0, 205, 100), 4: QColor(255, 255, 100),
    # Left side
    5: QColor(0, 200, 255), 7: QColor(0, 200, 255), 9: QColor(0, 200, 255),
    11: QColor(0, 255, 200), 13: QColor(0, 255, 200), 15: QColor(0, 255, 200),
    # Right side
    6: QColor(255, 128, 0), 8: QColor(255, 128, 0), 10: QColor(255, 128, 0),
    12: QColor(255, 50, 50), 14: QColor(255, 50, 50), 16: QColor(255, 50, 50)
}

CAMERA_KEYS = [
    "Camera1_M11139",
    "Camera2_M11140",
    "Camera3_M11141",
    "Camera4_M11458",
    "Camera5_M11459",
    "Camera6_M11461",
    "Camera7_M11462",
    "Camera8_M11463"
]
