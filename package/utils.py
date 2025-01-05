from ultralytics import YOLO
from supervision.detection.core import Detections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.gridspec as gridspec
import cv2

def get_model(model_path):
    return YOLO(model_path)

def get_best_detections(results):
    _name = None
    _class = None
    _confidence = None
    _boxes = None
    best_idx = 0
    best_score =0
    for enum, result in enumerate(results.summary()):
        box = list(result['box'].values())
        confidence = result['confidence']
        class_id = result['class']
        name = result['name']
        if confidence > best_score:
            best_score = confidence
            best_idx = enum
            _name = np.array([name])
            _class = np.array([class_id])
            _confidence = np.array([confidence])
            _boxes = np.array([box])
    return Detections(xyxy=_boxes, class_id=_class, confidence=_confidence, data={'class_name': _name})

def detect_consent_message(model_path, image):
    model = get_model(model_path)
    results = model.predict(image, verbose=False)[0]
    return get_best_detections(results)

def slice_image(image, bbox):
    x_min, y_min, x_max, y_max = map(int, bbox)
    crop = image[y_min:y_max, x_min:x_max]
    return crop

def get_label_bbox(image, label_path):
    H, W, _ = image.shape  # Get image height and width
    with open(label_path, 'r') as file:
        yolo_box = file.readlines()[0].split(" ")
        yolo_box = list(map(float, yolo_box))
    # Extract the YOLO box parameters
    class_id = yolo_box[0]
    x_center = yolo_box[1]
    y_center = yolo_box[2]
    width = yolo_box[3]
    height = yolo_box[4]

    # Convert YOLO format to pixel coordinates
    x_min = (x_center - width / 2) * W
    y_min = (y_center - height / 2) * H
    x_max = (x_center + width / 2) * W
    y_max = (y_center + height / 2) * H

    # Convert to integer coordinates for plotting
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    return x_min, y_min, x_max, y_max

def plot_result_of_three_model(image_path, model_list):
    img_path = image_path['image']
    label_path = image_path['label']
    image = cv2.imread(img_path)
    x_min, y_min, x_max, y_max = get_label_bbox(image, label_path)
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 1])
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    box = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
    ax1.add_patch(box)
    ax1.text(
        x_min, y_min-7, "consent", color='red', fontsize=12, ha='left', 
        # bbox=dict(facecolor='white', edgecolor='none')
        )
    ax1.set_title('Ground Truth')
    # ax1.axis('off')

    for enum, ax in enumerate([ax2, ax3, ax4]):
        _image = cv2.imread(img_path)
        detections = detect_consent_message(model_list[enum], _image)
        bbox = detections.xyxy[0]
        confidence = detections.confidence[0]
        _image = slice_image(_image, bbox)
        ax.imshow(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB))
        model_name = [m_name for m_name in model_list[enum].split('/') if "train_w_" in m_name][0]
        ax.set_title(f"Predict({confidence:.2f}) | {model_name}")
        # ax.axis('off')
    title = img_path.split("\\")[-1]

    fig.suptitle(f'{title}', fontsize=30)
    plt.tight_layout()
    plt.show()