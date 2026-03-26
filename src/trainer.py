from ultralytics import YOLO
from src.config import Config
import os

class AdvancedTrainer:
    def __init__(self, model_name=Config.MODEL_NAME):
        self.model_path = Config.get_model_path(model_name)
        # Load the model (downloads automatically if not found locally)
        self.model = YOLO(model_name) 
        
    def add_custom_callbacks(self):
        """Example of injecting custom logic into the training loop."""
        def on_train_epoch_end(trainer):
            print(f"--> Custom Log: Epoch {trainer.epoch + 1} finished.")
            
        self.model.add_callback("on_train_epoch_end", on_train_epoch_end)

    def train(self):
        print(f"Starting training for {Config.EPOCHS} epochs...")
        self.add_custom_callbacks()
        
        results = self.model.train(
            data=Config.DATA_YAML,
            epochs=Config.EPOCHS,
            imgsz=Config.IMG_SIZE,
            batch=Config.BATCH_SIZE,
            device=Config.DEVICE,
            project=Config.OUTPUT_DIR,
            name="yolo26_custom_run",
            exist_ok=True,
            # YOLO26 specific: Ensure NMS-free decoding is active if applicable
            # (Usually handled automatically by the cfg, but good to be aware)
        )
        print(f"Training complete. Results saved to {Config.OUTPUT_DIR}")
        return results

    def validate(self):
        metrics = self.model.val()
        print(f"mAP50: {metrics.box.map50}")
        return metrics