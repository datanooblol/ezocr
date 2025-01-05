import argparse
from ultralytics import YOLO
from glob import glob
import cv2
import os
import torch

def main(args):
    print("cuda available: ", torch.cuda.is_available())
    
    # Get dataset images path
    dataset = glob(os.path.join(args.dataset_path, "train", "images", "*"))
    
    # Read the first image to get its shape
    image = cv2.imread(dataset[0])
    h, w, c = image.shape
    
    # Load the YOLO model
    model = YOLO(args.model_path)
    
    # Use imgsz as a tuple
    img_size = tuple(args.imgsz)
    
    # Train the model
    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=img_size,
        workers=args.workers,
        device=args.device,
        verbose=args.verbose
    )
    
    # Remove cache labels from dataset
    for path in ["train", "val"]:
        os.remove(os.path.join(".", "dataset", path, "labels.cache"))

if __name__ == '__main__':
    # Argument parser setup
    parser = argparse.ArgumentParser(description="YOLO Model Training Script")
    
    parser.add_argument("--dataset_path", type=str, default="./dataset", help="Path to the dataset.")
    parser.add_argument("--model_path", type=str, default="yolo11m.pt", help="Path to the YOLO model.")
    parser.add_argument("--data_yaml", type=str, default="custom_dataset.yaml", help="Path to the dataset YAML file.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    
    # Modified --imgsz to accept multiple integers as a tuple
    parser.add_argument("--imgsz", type=int, nargs=2, default=(640, 640), help="Image size (tuple of two integers, e.g., --imgsz height width).")
    
    parser.add_argument("--workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--device", type=str, default="0", help="Device to train on (e.g., '0' for GPU).")
    parser.add_argument("--verbose", type=bool, default=False, help="Whether to print verbose output during training.")
    
    args = parser.parse_args()
    
    # Call main function with arguments
    main(args)

