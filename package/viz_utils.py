from supervision.detection.core import Detections
import cv2

def get_bboxes(results, best=False):
    best_idx = 0
    best_score = 0
    bboxes = []
    for idx, result in enumerate(results.summary()):
        score = result['confidence']
        if score > best_score:
            best_score = score
            best_idx = idx
        bbox = list(result['box'].values())
        bboxes.append(bbox)
    if best==True:
        return [bboxes[best_idx]]
    return bboxes

def draw_bboxes(image, bboxes):
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    return image

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