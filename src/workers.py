import os
from PyQt6.QtCore import QThread, pyqtSignal

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
    """Background computation thread to run YOLO and ViTPose on all images in the sequence."""
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, model_wrapper, images, img_ann_map):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.images = images  # list of image dicts
        self.img_ann_map = img_ann_map  # map of image_id -> annotation dict
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            # First initialize the models in this thread to make sure they are loaded
            self.model_wrapper.init_yolo()
            self.model_wrapper.init_vitpose()
            
            total = len(self.images)
            
            # Filter out images that are already processed
            images_to_process = []
            skipped_count = 0
            
            for idx, img_entry in enumerate(self.images):
                img_id = img_entry["id"]
                ann = self.img_ann_map[img_id]
                if ann.get("bbox") and sum(ann["bbox"]) > 0:
                    skipped_count += 1
                    self.progress.emit(idx + 1, total, f"Saut de {os.path.basename(img_entry['file_name'])} (déjà traité)")
                else:
                    images_to_process.append((idx, img_entry))
            
            processed_count = 0
            if images_to_process:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                
                # Sweet spot for max_workers: 4 workers
                # This balances OpenCV I/O, resizing and PyTorch inference without overloading GPU or CPU
                max_workers = 8
                
                def process_image(idx, img_entry):
                    if self._is_cancelled:
                        return idx, img_entry, None, None
                    try:
                        image_path = img_entry["file_name"]
                        bbox = self.model_wrapper.run_yolo(image_path)
                        keypoints = None
                        if bbox and not self._is_cancelled:
                            keypoints = self.model_wrapper.run_vitpose(image_path, bbox)
                        return idx, img_entry, bbox, keypoints
                    except Exception as e:
                        return idx, img_entry, e, None

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(process_image, idx, img_entry): (idx, img_entry)
                        for idx, img_entry in images_to_process
                    }
                    
                    completed_tasks_count = skipped_count
                    for future in as_completed(futures):
                        if self._is_cancelled:
                            # Cancel pending futures
                            for f in futures:
                                f.cancel()
                            break
                            
                        try:
                            idx, img_entry, bbox, keypoints = future.result()
                            
                            # If an exception was raised during task execution, propagate it
                            if isinstance(bbox, Exception):
                                raise bbox
                                
                            if bbox:
                                img_id = img_entry["id"]
                                ann = self.img_ann_map[img_id]
                                ann["bbox"] = bbox
                                if keypoints:
                                    flat_kps = []
                                    for kp in keypoints:
                                        flat_kps.extend(kp)
                                    ann["keypoints"] = flat_kps
                                    ann["num_keypoints"] = sum(1 for idx_kp in range(17) if flat_kps[idx_kp*3 + 2] > 0)
                                processed_count += 1
                                
                            completed_tasks_count += 1
                            self.progress.emit(completed_tasks_count, total, f"Traitement de {os.path.basename(img_entry['file_name'])}...")
                            
                        except Exception as e:
                            # Cancel remaining tasks and raise
                            for f in futures:
                                f.cancel()
                            raise e
            
            self.finished.emit(processed_count)
        except Exception as e:
            self.error.emit(str(e))
