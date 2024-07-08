from typing import List, Tuple, Union
import cv2
import random
from numpy import ndarray
import numpy as np
import configparser


def non_max_suppression(boxes: Union[List[ndarray], Tuple[ndarray]],
                        scores: Union[List[float], Tuple[float]],
                        labels: Union[List[int], Tuple[int]],
                        conf_thres: float = 0.25,
                        iou_thres: float = 0.65) -> Tuple[List, List, List]:

    MAJOR, MINOR = map(int, cv2.__version__.split('.')[:2])
    assert MAJOR == 4
    if MINOR >= 7:
        indices = cv2.dnn.NMSBoxesBatched(boxes, scores, labels, conf_thres,
                                          iou_thres)
    elif MINOR == 6:
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    else:
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres,
                                   iou_thres).flatten()

    nmsd_boxes = []
    nmsd_scores = []
    nmsd_labels = []
    for idx in indices:
        box = boxes[idx]
        # x0y0wh -> x0y0x1y1
        box[2:] = box[:2] + box[2:]
        score = scores[idx]
        label = labels[idx]
        nmsd_boxes.append(box)
        nmsd_scores.append(score)
        nmsd_labels.append(label)

    return nmsd_boxes, nmsd_scores, nmsd_labels


def softmax(x: ndarray, axis: int = -1) -> ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    y = e_x / e_x.sum(axis=axis, keepdims=True)

    return y


def sigmoid(x: ndarray) -> ndarray:

    return 1. / (1. + np.exp(-x))


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if left != 0 or top != 0 or bottom != 0 or right != 0:
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        img = img.astype(np.uint8)

    return img, np.array(ratio), np.array([dw, dh])


def read_config(cfg_path: str):
    config = configparser.ConfigParser()
    config.read(cfg_path)

    return config


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1]), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_detect_results(output, img, is_show_label=True, color=[0, 165, 255]):
    if len(output) > 0:
        for element in output:
            xyxy = element['xyxy']
            cls = element['name']
            conf = element['conf']
            label = None
            if is_show_label:
                label = f'{cls} {conf:.2f}'

            plot_one_box(xyxy, img, color, label)


def read_img(img_path):
    img = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(img, flags=cv2.IMREAD_COLOR)

    return img


def save_img(img, img_path):
    img_suffix = img_path.split('.')[-1]
    cv2.imencode('.{}'.format(img_suffix), img)[1].tofile(img_path)
