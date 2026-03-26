"""Utility functions for YOLO26 project"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
from src.config import LOGGING_CONFIG

# Configure logging
logging.basicConfig(
    level=LOGGING_CONFIG["level"],
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)


def setup_logger(name: str) -> logging.Logger:
    """Setup logger for a module"""
    return logging.getLogger(name)


def load_image(image_path: str) -> np.ndarray:
    """Load an image from file"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        return image
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise


def load_video(video_path: str) -> cv2.VideoCapture:
    """Load a video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        return cap
    except Exception as e:
        logger.error(f"Error loading video: {e}")
        raise


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save an image to file"""
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)
        logger.info(f"Image saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        raise


def draw_detections(
    image: np.ndarray, 
    detections: List[Dict[str, Any]], 
    class_names: List[str] = None
) -> np.ndarray:
    """Draw bounding boxes and labels on image"""
    try:
        result = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["bbox"])
            confidence = detection.get("confidence", 0)
            class_id = detection.get("class_id", 0)
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_names[class_id] if class_names else f'Class {class_id}'}: {confidence:.2f}"
            cv2.putText(
                result, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        return result
    except Exception as e:
        logger.error(f"Error drawing detections: {e}")
        raise


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size"""
    try:
        resized = cv2.resize(image, target_size)
        return resized
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        raise


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range"""
    try:
        normalized = image.astype(np.float32) / 255.0
        return normalized
    except Exception as e:
        logger.error(f"Error normalizing image: {e}")
        raise


def create_data_splits(
    data_dir: Path, 
    split_ratio: Dict[str, float] = None
) -> Dict[str, List[str]]:
    """Create train/val/test splits from data directory"""
    if split_ratio is None:
        split_ratio = {"train": 0.7, "val": 0.2, "test": 0.1}
    
    try:
        all_files = list(data_dir.glob("*"))
        np.random.shuffle(all_files)
        
        total = len(all_files)
        train_split = int(total * split_ratio["train"])
        val_split = int(total * split_ratio["val"])
        
        splits = {
            "train": all_files[:train_split],
            "val": all_files[train_split:train_split + val_split],
            "test": all_files[train_split + val_split:]
        }
        
        logger.info(f"Data splits created: {[len(v) for v in splits.values()]}")
        return splits
    except Exception as e:
        logger.error(f"Error creating data splits: {e}")
        raise
