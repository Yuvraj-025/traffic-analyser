import argparse
from src.trainer import AdvancedTrainer
from src.inference import InferenceEngine

def main():
    parser = argparse.ArgumentParser(description="YOLO26 Advanced Project Runner")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'], 
                        help="Choose 'train' to fine-tune or 'predict' for inference")
    parser.add_argument('--source', type=str, default='0', 
                        help="Path to video file or '0' for webcam (for predict mode)")
    parser.add_argument('--model', type=str, default=None, 
                        help="Path to custom model .pt file (optional)")

    args = parser.parse_args()

    if args.mode == 'train':
        trainer = AdvancedTrainer()
        trainer.train()
    elif args.mode == 'predict':
        engine = InferenceEngine(model_path=args.model)
        # Convert source to int if it's a digit (for webcam index)
        src = int(args.source) if args.source.isdigit() else args.source
        engine.run_live(source=src)

if __name__ == "__main__":
    main()