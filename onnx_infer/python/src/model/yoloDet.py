from ..engine import Preprocess, Decoder
from numpy import ndarray
from ..utils import non_max_suppression, read_config
import onnxruntime
import numpy as np
import math
import ast


class YOLODet:
    def __init__(self, config_path: str = ''):
        self.config = read_config(config_path)
        self.model_section = self.config.sections()[0]
        if len(self.model_section) > 1:
            assert 'Only one model is supported at a time. Please check the model name in the config file.'
        self.model_type = self.model_section.upper()
        self.onnx_path = self.config.get(self.model_section, 'onnx_path')
        self.gpu_id = int(self.config.get(self.model_section, 'gpu_id'))
        self.model_only = bool(self.config.get(
            self.model_section, 'model_only'))
        self.score_thr = float(self.config.get(
            self.model_section, 'score_thr'))
        self.iou_thr = float(self.config.get(self.model_section, 'iou_thr'))
        self.class_name = ast.literal_eval(
            self.config.get(self.model_section, 'class_name'))
        self.filter_class_name = ast.literal_eval(self.config.get(
            self.model_section, 'filter_class_name'))
        self.inference_size = ast.literal_eval(self.config.get(
            self.model_section, 'inference_size'))
        self.is_letterbox = bool(self.config.get(
            self.model_section, 'is_letterbox'))
        if self.config.has_option(self.model_section, 'yolov5_anchors'):
            self.anchors = ast.literal_eval(self.config.get(
                self.model_section, 'yolov5_anchors'))
        elif self.config.has_option(self.model_section, 'yolov7_anchors'):
            self.anchors = ast.literal_eval(self.config.get(
                self.model_section, 'yolov7_anchors'))
        self.session = None
        self.preprocessor = None
        self.decoder = None

    def onnx_init(self):
        if self.onnx_path is None:
            assert 'Please provide the path to the onnx model'

        if self.gpu_id is None or self.gpu_id < 0:
            providers = ['CPUExecutionProvider']
        else:
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': self.gpu_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider',
            ]

        self.session = onnxruntime.InferenceSession(
            self.onnx_path, providers=providers)
        self.preprocessor = Preprocess(self.model_type)
        self.decoder = Decoder(self.model_type, model_only=self.model_only)

    def onnx_infer(self, img: ndarray):

        origin_image_w, origin_image_h = img.shape[:2]
        resize_img, (ratio_w, ratio_h), dxdy = self.preprocessor(
            img, self.inference_size, self.is_letterbox)
        features = self.session.run(
            None, {self.session.get_inputs()[0].name: resize_img})
        output = self.onnx_postprocess(features, self.decoder, self.class_name, self.filter_class_name, self.score_thr, self.iou_thr,
                                       ratio_w, ratio_h, dxdy, origin_image_w, origin_image_h, self.anchors)

        return output

    def onnx_postprocess(self, features, decoder, class_name, filter_class_name, score_thr, iou_thr,
                         ratio_w, ratio_h, dxdy, image_w, image_h, anchors=None):
        decoder_outputs = decoder(
            features,
            score_thr,
            num_labels=len(class_name),
            anchors=anchors)

        nmsd_boxes, nmsd_scores, nmsd_labels = non_max_suppression(
            *decoder_outputs, score_thr, iou_thr)

        anns = []
        for box, score, label in zip(nmsd_boxes, nmsd_scores, nmsd_labels):
            if class_name[label] not in filter_class_name:
                continue
            x0, y0, x1, y1 = box
            if dxdy is not None:
                x0 = (x0 - dxdy[0]) / ratio_w
                y0 = (y0 - dxdy[1]) / ratio_h
                x1 = (x1 - dxdy[0]) / ratio_w
                y1 = (y1 - dxdy[1]) / ratio_h
            else:
                x0 = math.floor(min(max(x0 / ratio_w, 1), image_w - 1))
                y0 = math.floor(min(max(y0 / ratio_h, 1), image_h - 1))
                x1 = math.ceil(min(max(x1 / ratio_w, 1), image_w - 1))
                y1 = math.ceil(min(max(y1 / ratio_h, 1), image_h - 1))

            ann = {'xyxy': np.array([x0, y0, x1, y1]),
                   'name': class_name[label], 'conf': score}
            anns.append(ann)

        return anns
