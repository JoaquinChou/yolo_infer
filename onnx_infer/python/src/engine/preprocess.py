from typing import List, Tuple, Union
from numpy import ndarray
from ..utils import letterbox
import cv2
import numpy as np


class Preprocess:

    def __init__(self, model_type: str, img_mean: List, img_std: List, img_is_rgb: bool):
        print(f"model_type: {model_type}")
        if model_type in ("YOLOV5", "YOLOX", "YOLOV6", "YOLOV7", "YOLOV8", "PPYOLOE", "PPYOLOEP", "RTMDET"):
            mean = np.array(img_mean, dtype=np.float32)
            std = np.array(img_std, dtype=np.float32)
            is_rgb = img_is_rgb
        else:
            raise NotImplementedError('Only supported model type:{"YOLOV5", "YOLOX", \
                "PPYOLOE", "PPYOLOEP", "YOLOV6", "YOLOV7", "RTMDET", "YOLOV8"}')

        self.mean = mean.reshape((3, 1, 1))
        self.std = std.reshape((3, 1, 1))
        self.is_rgb = is_rgb

    def __call__(self,
                 image: ndarray,
                 new_size: Union[List[int], Tuple[int]] = (640, 640),
                 is_letterbox: bool = False,
                 **kwargs) -> Tuple[ndarray, Tuple[float, float]]:

        dxdy = None
        if is_letterbox:
            image, ratio, dxdy = letterbox(
                image, new_shape=new_size, color=(114, 114, 114))
            ratio_w, ratio_h = ratio
            image = np.ascontiguousarray(image.transpose(2, 0, 1))
            image = image.astype(np.float32)
            image -= self.mean
            image /= self.std

        else:
            height, width = image.shape[:2]
            ratio_h, ratio_w = new_size[0] / height, new_size[1] / width
            image = cv2.resize(
                image, (0, 0),
                fx=ratio_w,
                fy=ratio_h,
                interpolation=cv2.INTER_LINEAR)

        return image[np.newaxis], (ratio_w, ratio_h), dxdy
