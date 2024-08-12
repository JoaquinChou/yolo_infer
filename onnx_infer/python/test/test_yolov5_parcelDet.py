import cv2
import sys
sys.path.append('../')
from src.model import YOLODet
from src.utils import plot_detect_results, save_img


def main():
    config_path = '../config/YOLOV5_bind_1c.ini'
    test_img_path = './data/test_img/ring_parcel.jpg'
    save_img_path = './data/test_res/ring_parcel.jpg'
    test_img = cv2.imread(test_img_path)
    yolov5Det = YOLODet(config_path)
    yolov5Det.onnx_init()
    output = yolov5Det.onnx_infer(test_img)

    print("output->", output)
    plot_detect_results(output, test_img, is_show_label=False)
    save_img(test_img, save_img_path)

if __name__ == "__main__":
    main()
