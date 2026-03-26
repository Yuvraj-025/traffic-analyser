"""
auto_start.py — Quick training launcher (moved to scripts/)
Run from project ROOT:  python scripts/auto_start.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

def main():
    model = YOLO('yolo26n.pt')

    print("Downloading VisDrone dataset and starting training...")
    print("Note: This dataset is large (~6 GB). Ensure you have a good connection.")

    results = model.train(
        data='VisDrone.yaml',
        epochs=10,
        imgsz=640,
        batch=16,
        device='cpu',
        project='runs/train',
        name='yolo26_visdrone'
    )

    print("Training complete! Best model saved in runs/train/yolo26_visdrone/weights/best.pt")

if __name__ == '__main__':
    main()