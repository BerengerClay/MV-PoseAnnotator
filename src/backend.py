import os
import cv2
import numpy as np
import torch
import threading
from src.vitpose_model import load_vitpose_model
from ultralytics import YOLO

class ModelWrapper:
    """Wrapper class to handle initialization and inference of YOLO and ViTPose models."""
    def __init__(self, weights_dir=None, device="cpu"):
        self.device = device
        
        # Resolve weights directory relative to root directory if not specified
        if weights_dir is None or weights_dir == "weights":
            src_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(src_dir)
            self.weights_dir = os.path.join(root_dir, "weights")
        else:
            self.weights_dir = weights_dir
            
        self._local = threading.local()
        self.yolo_model = None
        self.vitpose_model = None

    @property
    def yolo_model(self):
        if not hasattr(self._local, "yolo_model"):
            self._local.yolo_model = None
        return self._local.yolo_model

    @yolo_model.setter
    def yolo_model(self, val):
        self._local.yolo_model = val

    def init_yolo(self):
        """Initializes the YOLO object detector, falling back to a PyTorch model if engine fails or CUDA is unavailable."""
        if self.yolo_model is not None:
            return
            
        engine_path = os.path.join(self.weights_dir, "YOLO26s_best.engine")
        
        # TensorRT engine files can ONLY run on CUDA. If CUDA is not available, we must fall back to PyTorch .pt
        cuda_available = False
        try:
            cuda_available = torch.cuda.is_available()
        except Exception:
            pass
            
        if os.path.exists(engine_path) and cuda_available:
            try:
                # Also verify if tensorrt package is installed
                import tensorrt
                print(f"Loading YOLO TensorRT engine from {engine_path}...")
                self.yolo_model = YOLO(engine_path)
                print("YOLO TensorRT engine loaded successfully.")
                return
            except Exception as e:
                print(f"TensorRT engine load/verification failed: {e}. Falling back to standard PyTorch model.")
                
        # Fallback to PyTorch model
        pt_path = os.path.join(self.weights_dir, "YOLO26s_best.pt")
        if not os.path.exists(pt_path):
            pt_path = os.path.join(self.weights_dir, "yolov8s.pt")
            
        print(f"Loading YOLO PyTorch model from {pt_path}...")
        try:
            self.yolo_model = YOLO(pt_path if os.path.exists(pt_path) else "yolov8s.pt")
            print(f"YOLO PyTorch model ({os.path.basename(pt_path)}) loaded successfully.")
        except Exception as ex:
            print(f"Could not load fallback YOLO: {ex}")
            raise ex

    def init_vitpose(self):
        """Initializes ViTPose-s pose estimator."""
        if self.vitpose_model is not None:
            return
            
        pth_path = os.path.join(self.weights_dir, "best_ViTPose-s_AP731.pth")
        if not os.path.exists(pth_path):
            raise FileNotFoundError(f"ViTPose weights not found at: {pth_path}")
            
        try:
            print(f"Loading ViTPose PyTorch model from {pth_path}...")
            self.vitpose_model = load_vitpose_model(pth_path, device=self.device)
            print("ViTPose model loaded successfully.")
        except Exception as e:
            print(f"Failed to load ViTPose: {e}")
            raise e

    def run_yolo(self, image_path):
        """Detects the jumper bounding box [x, y, w, h]."""
        self.init_yolo()
        
        # Run YOLO detector
        # conf=0.25, classes=[0] to focus on person (trampoline jumper)
        results = self.yolo_model(image_path, verbose=False, conf=0.25)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            # Return the bounding box with the highest confidence
            best_idx = int(boxes.conf.argmax())
            xyxy = boxes.xyxy[best_idx].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            
        return None

    def run_vitpose(self, image_path, bbox):
        """Runs ViTPose on the cropped bounding box to get 17 COCO 2D keypoints."""
        self.init_vitpose()
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        h_orig, w_orig = img.shape[:2]
        x, y, w, h = bbox
        
        # Clamp crop coordinates to image boundary
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w_orig, int(x + w)), min(h_orig, int(y + h))
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        # Crop jumper
        crop = img[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape[:2]
        
        # Preprocess crop: resize to (192, 256) [W, H], normalize, convert to tensor
        crop_resized = cv2.resize(crop, (192, 256))
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        
        # PIL/timm normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        tensor = torch.from_numpy(crop_rgb).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Model forward pass
        with torch.no_grad():
            heatmaps = self.vitpose_model(tensor)
            
        # heatmaps: (1, 17, 64, 48) [B, joints, H_hm, W_hm]
        heatmaps = heatmaps.squeeze(0).cpu().numpy()
        
        # Extract peaks of heatmaps
        keypoints = []
        for i in range(17):
            hm = heatmaps[i]
            # Get argmax index
            idx = hm.argmax()
            y_hm, x_hm = np.unravel_index(idx, hm.shape)
            conf = float(hm[y_hm, x_hm])
            
            # Map back to crop coordinates (upsample from 64x48 to 256x192)
            # 256 / 64 = 4.0, 192 / 48 = 4.0
            x_crop = (x_hm + 0.5) * 4.0
            y_crop = (y_hm + 0.5) * 4.0
            
            # Map crop coordinates back to original image
            x_orig = x1 + (x_crop / 192.0) * crop_w
            y_orig = y1 + (y_crop / 256.0) * crop_h
            
            # Visibility: 2 = Manual/Confirmed, 1 = Estimated/Low Conf (we mark as 2 by default so user can edit directly)
            keypoints.append([x_orig, y_orig, 2])
            
        return keypoints


def triangulate_and_reproject(keypoints_data, camera_matrices):
    """
    Performs 3D Triangulation using SVD on keypoints labeled on 2+ cameras, 
    and reprojects the resulting 3D coordinates onto non-annotated cameras.
    
    Args:
        keypoints_data (dict): {cam_id: [[x, y, v], ...]} (17 keypoints per camera)
        camera_matrices (list): list of 8 projection matrices of shape (3x4)
        
        
    Returns:
        dict: updated keypoints data with projected estimations
    """
    updated_keypoints = {cam_id: [kp[:] for kp in kps] for cam_id, kps in keypoints_data.items()}
    
    for kp_idx in range(17):
        # 1. Identify cameras that have visible keypoints (v > 0) to use as the triangulation base
        base_cams = []
        for cam_id in range(8):
            kp = keypoints_data[cam_id][kp_idx]
            if kp[2] > 0:  # Any visible keypoint (reprojected or manual)
                base_cams.append(cam_id)
                
        if len(base_cams) < 2:
            # Need at least 2 camera views to triangulate 3D coordinates
            continue
            
        # 2. Build SVD equation matrix A
        A = []
        for cam_id in base_cams:
            P = np.array(camera_matrices[cam_id]) # 3x4 projection matrix
            u, v, _ = keypoints_data[cam_id][kp_idx]
            A.append(u * P[2, :] - P[0, :])
            A.append(v * P[2, :] - P[1, :])
            
        A = np.array(A)
        
        # 3. Solve AX = 0 via SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1, :] # Last row of Vt is the right singular vector corresponding to smallest singular value
        
        if X[3] != 0:
            X = X / X[3] # Normalize homogeneous coordinate
            X_3d = X[:3]
            
            # Upgrade all cameras used as base to manual reference visibility (v = 2)
            for cam_id in base_cams:
                u, v, _ = keypoints_data[cam_id][kp_idx]
                updated_keypoints[cam_id][kp_idx] = [float(u), float(v), 2]
            
            # 4. Project the 3D homogeneous point back to all other cameras
            for cam_id in range(8):
                if cam_id not in base_cams:
                    P = np.array(camera_matrices[cam_id])
                    X_homog = np.array([X_3d[0], X_3d[1], X_3d[2], 1.0])
                    x_proj = P @ X_homog
                    
                    if x_proj[2] != 0:
                        u_proj = x_proj[0] / x_proj[2]
                        v_proj = x_proj[1] / x_proj[2]
                        
                        # Only project if the coordinate is within the 1920x1080 image boundary
                        if 0.0 <= u_proj <= 1920.0 and 0.0 <= v_proj <= 1080.0:
                            current_kp = keypoints_data[cam_id][kp_idx]
                            # Only project if the keypoint is currently absent (v == 0)
                            if current_kp[2] == 0:
                                updated_keypoints[cam_id][kp_idx] = [float(u_proj), float(v_proj), 1]
                                
    return updated_keypoints
