from PyQt6.QtCore import QThread, pyqtSignal
from src.constants import CAMERA_KEYS

class WorkerThread(QThread):
    """Background computation thread to run YOLO and ViTPose without freezing UI."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, task_type, model_wrapper, args):
        super().__init__()
        self.task_type = task_type
        self.model_wrapper = model_wrapper
        self.args = args

    def run(self):
        try:
            if self.task_type == "yolo_vitpose":
                image_path = self.args["image_path"]
                camera_id = self.args["camera_id"]
                # 1. Run YOLO to get bbox
                bbox = self.model_wrapper.run_yolo(image_path)
                # 2. Run ViTPose on the detected bbox
                keypoints = None
                if bbox:
                    keypoints = self.model_wrapper.run_vitpose(image_path, bbox)
                self.finished.emit({
                    "camera_id": camera_id,
                    "bbox": bbox,
                    "keypoints": keypoints
                })
            elif self.task_type == "vitpose_only":
                image_path = self.args["image_path"]
                camera_id = self.args["camera_id"]
                bbox = self.args["bbox"]
                keypoints = self.model_wrapper.run_vitpose(image_path, bbox)
                self.finished.emit({
                    "camera_id": camera_id,
                    "bbox": bbox,
                    "keypoints": keypoints
                })
        except Exception as e:
            self.error.emit(str(e))


class SequencePreprocessWorker(QThread):
    """Background computation thread to run YOLO and ViTPose on all images in the sequence using batching."""
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, model_wrapper, sorted_frames, frame_data, img_file_map, img_ann_map):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.sorted_frames = sorted_frames
        self.frame_data = frame_data
        self.img_file_map = img_file_map
        self.img_ann_map = img_ann_map
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            # First initialize the models to make sure they are loaded
            self.model_wrapper.init_yolo()
            self.model_wrapper.init_vitpose()
            
            total_frames = len(self.sorted_frames)
            processed_images_count = 0
            
            for f_idx, frame_idx in enumerate(self.sorted_frames):
                if self._is_cancelled:
                    break
                
                # Group views of this frame that need to be processed
                images_to_process = []  # list of (cam_key, path, img_id, ann)
                
                for cam_key in CAMERA_KEYS:
                    path = self.frame_data[frame_idx].get(cam_key)
                    if path:
                        img_entry = self.img_file_map.get(path)
                        if img_entry:
                            img_id = img_entry["id"]
                            ann = self.img_ann_map.get(img_id)
                            # Only process if bbox is not drawn yet
                            if ann and (not ann.get("bbox") or sum(ann["bbox"]) == 0):
                                images_to_process.append((cam_key, path, img_id, ann))
                
                if not images_to_process:
                    # All cameras for this frame are already processed
                    self.progress.emit(f_idx + 1, total_frames, f"Frame {f_idx + 1}/{total_frames} déjà traitée")
                    continue
                
                # Extract image paths for batch processing
                paths = [item[1] for item in images_to_process]
                
                # 1. Run YOLO batch on the images
                bboxes = self.model_wrapper.run_yolo_batch(paths)
                
                # 2. Run ViTPose batch on the images
                keypoints_list = self.model_wrapper.run_vitpose_batch(paths, bboxes)
                
                # 3. Save predictions back to memory database
                for idx, (cam_key, path, img_id, ann) in enumerate(images_to_process):
                    bbox = bboxes[idx]
                    keypoints = keypoints_list[idx]
                    
                    if bbox:
                        ann["bbox"] = bbox
                        if keypoints:
                            flat_kps = []
                            for kp in keypoints:
                                flat_kps.extend(kp)
                            ann["keypoints"] = flat_kps
                            ann["num_keypoints"] = sum(1 for idx_kp in range(17) if flat_kps[idx_kp*3 + 2] > 0)
                        processed_images_count += 1
                
                self.progress.emit(f_idx + 1, total_frames, f"Traitement de la frame {f_idx + 1}/{total_frames}...")
            
            self.finished.emit(processed_images_count)
        except Exception as e:
            self.error.emit(str(e))
