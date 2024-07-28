import cv2
import sys
sys.path.append('../')
from src.model import YOLODet
from src.utils import plot_detect_results, save_img


def main():
    config_path = '../config/YOLOV8_coco_80c.ini'
    test_img_path = './data/test_img/COCO_train2014_000000003157.jpg'
    save_img_path = './data/test_res/COCO_train2014_000000003157.jpg'
    test_img = cv2.imread(test_img_path)
    yolov8Det = YOLODet(config_path)
    yolov8Det.onnx_init()
    output = yolov8Det.onnx_infer(test_img)

    print("output->", output)
    plot_detect_results(output, test_img, is_show_label=True)
    save_img(test_img, save_img_path)

if __name__ == "__main__":
    main()
