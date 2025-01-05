from ultralytics import YOLO
from glob import glob
import cv2
import os
import torch

def main():
    print("cuda available: ", torch.cuda.is_available())
    dataset_path = "./dataset/train/images/*"
    dataset = glob(dataset_path)
    image = cv2.imread(dataset[0])
    h, w, c = image.shape
    model = YOLO("yolo11m.pt")
    model.train(
        data="custom_dataset.yaml",
        epochs=100,
        batch=16,
        # imgsz=(h, w),
        imgsz=640,
        workers=0,
        device="0",
        verbose=False
    )
    for path in ["train", "val"]:
        os.remove(os.path.join(".", "dataset", path, "labels.cache"))

if __name__ == '__main__':
    main()