import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_YAML = os.path.join(BASE_DIR, 'data', 'dataset.yaml') # Point to your data.yaml
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'runs')
    
    # Model Settings
    MODEL_NAME = "yolo26n.pt"  # Options: yolo26n, yolo26s, yolo26m, etc.
    IMG_SIZE = 640
    CONF_THRESHOLD = 0.5
    
    # Training Hyperparameters
    EPOCHS = 50
    BATCH_SIZE = 16
    DEVICE = "0"  # GPU ID or 'cpu'

    @staticmethod
    def get_model_path(model_name):
        return os.path.join(Config.MODEL_DIR, model_name)