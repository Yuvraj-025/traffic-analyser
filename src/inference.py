import cv2
from ultralytics import YOLO
from src.config import Config
import time

class InferenceEngine:
    def __init__(self, model_path=None):
        # Load custom trained model if provided, else base model
        load_path = model_path if model_path else Config.get_model_path(Config.MODEL_NAME)
        self.model = YOLO(load_path)
        print(f"Loaded model from: {load_path}")

    def run_live(self, source=0):
        """Run inference on webcam (0) or video file path."""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise ValueError("Could not open video source.")

        print("Press 'q' to exit.")
        
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO26 Inference
            # stream=True is efficient for video memory
            results = self.model(frame, conf=Config.CONF_THRESHOLD, stream=True)

            # Process results
            for result in results:
                # Plot the detections directly onto the frame
                annotated_frame = result.plot() 
                
                # Calculate FPS
                fps = 1.0 / (time.time() - start_time)
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("YOLO26 Advanced Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()