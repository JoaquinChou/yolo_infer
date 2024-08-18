from typing import List, Tuple, Union
from ..utils import sigmoid, softmax
from numpy import ndarray
import numpy as np


class ULTRAL_Decoder:

    def __init__(self, model_type: str = '', model_only: bool = False):
        self.model_type = model_type
        self.model_only = model_only
        self.boxes_pro = []
        self.scores_pro = []
        self.labels_pro = []

    def __call__(self,
                 feats: Union[List, Tuple],
                 conf_thres: float,
                 num_labels: int = 80,
                 **kwargs) -> Tuple:

        self.boxes_pro.clear()
        self.scores_pro.clear()
        self.labels_pro.clear()

        feats = [
            np.ascontiguousarray(feat[0].transpose(1, 0))
            for feat in feats
        ]

        if self.model_type == "YOLOV5":
            self.__yolov5_decode(feats, conf_thres, num_labels, **kwargs)
        elif self.model_type == "YOLOX":
            self.__yolox_decode(feats, conf_thres, num_labels, **kwargs)
        elif self.model_type in ("PPYOLOE", "PPYOLOEP"):
            self.__ppyoloe_decode(feats, conf_thres, num_labels, **kwargs)
        elif self.model_type == "YOLOV6":
            self.__yolov6_decode(feats, conf_thres, num_labels, **kwargs)
        elif self.model_type == "YOLOV7":
            self.__yolov7_decode(feats, conf_thres, num_labels, **kwargs)
        elif self.model_type == "RTMDET":
            self.__rtmdet_decode(feats, conf_thres, num_labels, **kwargs)
        elif self.model_type == "YOLOV8":
            self.__yolov8_decode(feats, conf_thres, num_labels, **kwargs)
        else:
            raise NotImplementedError('Only supported model type:{"YOLOV5", "YOLOX", \
                "PPYOLOE", "PPYOLOEP", "YOLOV6", "YOLOV7", "RTMDET", "YOLOV8"}')
        return self.boxes_pro, self.scores_pro, self.labels_pro

    def __yolov5_decode(self,
                        feats: List[ndarray],
                        conf_thres: float,
                        num_labels: int = 80,
                        **kwargs):
        anchors: Union[List, Tuple] = kwargs.get(
            'anchors',
            [[(10, 13), (16, 30),
              (33, 23)], [(30, 61), (62, 45),
                          (59, 119)], [(116, 90), (156, 198), (373, 326)]])
        
        # YOLOv5u adopts an anchor-free split Ultralytics head.
        self.__yolov8_decode(feats, conf_thres, num_labels)

    def __yolox_decode(self,
                       feats: List[ndarray],
                       conf_thres: float,
                       num_labels: int = 80,
                       **kwargs):
        pass

    def __ppyoloe_decode(self,
                         feats: List[ndarray],
                         conf_thres: float,
                         num_labels: int = 80):

        for _, feat in enumerate(feats):
            _, box_with_cls_size = feat.shape
            box_feat, score_feat = np.split(
                feat, [box_with_cls_size - num_labels], -1)
            _max = np.max(score_feat, axis=-1)
            _argmax = score_feat.argmax(-1)
            proposal = np.where(_max > conf_thres)[0]
            num_proposal = proposal.size

            scores = _max[proposal]
            labels = _argmax[proposal]
            boxes = box_feat[proposal]

            for k in range(num_proposal):
                score = scores[k]
                label = labels[k]
                x0, y0, x1, y1 = boxes[k]

                x0 = x0 - x1 / 2
                y0 = y0 - y1 / 2
                w = x1
                h = y1

                self.scores_pro.append(float(score))
                self.boxes_pro.append(
                    np.array([x0, y0, w, h], dtype=np.float32))
                self.labels_pro.append(int(label))

    def __yolov6_decode(self,
                        feats: List[ndarray],
                        conf_thres: float,
                        num_labels: int = 80,
                        **kwargs):
        pass

    def __yolov7_decode(self,
                        feats: List[ndarray],
                        conf_thres: float,
                        num_labels: int = 80,
                        **kwargs):
        anchors: Union[List, Tuple] = kwargs.get(
            'anchors',
            [[(12, 16), (19, 36),
              (40, 28)], [(36, 75), (76, 55),
                          (72, 146)], [(142, 110), (192, 243), (459, 401)]])
        self.__yolov5_decode(feats, conf_thres, num_labels, anchors=anchors)

    def __rtmdet_decode(self,
                        feats: List[ndarray],
                        conf_thres: float,
                        num_labels: int = 80,
                        **kwargs):
        pass

    def __yolov8_decode(self,
                        feats: List[ndarray],
                        conf_thres: float,
                        num_labels: int = 80,
                        **kwargs):
        self.__ppyoloe_decode(feats, conf_thres, num_labels)
