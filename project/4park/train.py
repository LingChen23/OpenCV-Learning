from __future__ import division
import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from Parking import Parking
import pickle

cwd = os.getcwd()


def img_process(test_images, park):
    white_yellow_images = list(map(park.select_rgb_white_yellow, test_images))
    park.show_images(white_yellow_images)

    gray_images = list(map(park.convert_gray_scale, white_yellow_images))
    park.show_images(gray_images)

    edge_images = list(map(lambda image: park.detect_edges(image), gray_images))
    park.show_images(edge_images)

    roi_images = list(map(park.select_region, edge_images))
    park.show_images(roi_images)

    list_of_lines = list(map(park.hough_lines, roi_images))

    line_images = []
    for image, lines in zip(test_images, list_of_lines):
        line_images.append(park.draw_lines(image, lines))
    park.show_images(line_images)

    rect_images = []
    rect_coords = []
    for image, lines in zip(test_images, list_of_lines):
        new_image, rects = park.identify_blocks(image, lines)
        rect_images.append(new_image)
        rect_coords.append(rects)

    park.show_images(rect_images)

    delineated = []
    spot_pos = []
    for image, rects in zip(test_images, rect_coords):
        new_image, spot_dict = park.draw_parking(image, rects)
        delineated.append(new_image)
        spot_pos.append(spot_dict)

    park.show_images(delineated)
    final_spot_dict = spot_pos[1]
    print(len(final_spot_dict))

    with open('spot_dict.pickle', 'wb') as handle:
        pickle.dump(final_spot_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    park.save_images_for_cnn(test_images[0], final_spot_dict)

    return final_spot_dict


def keras_model(weights_path):
    model = load_model(weights_path)
    return model


def img_test(test_images, final_spot_dict, model, class_dictionary, park):
    for i in range(len(test_images)):
        predicted_images = park.predict_on_image(test_images[i], final_spot_dict, model, class_dictionary)


def video_test(video_name, final_spot_dict, model, class_dictionary, park):
    if not os.path.exists(video_name):
        raise FileNotFoundError(f"视频文件 {video_name} 不存在")

    cap = cv2.VideoCapture(video_name)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件 {video_name}")

    park.predict_on_video(video_name, final_spot_dict, model, class_dictionary, ret=True)
    cap.release()


if __name__ == '__main__':
    # 检查路径
    if not os.path.exists('test_images'):
        raise FileNotFoundError("test_images目录不存在")

    test_images = []
    for path in glob.glob('test_images/*.jpg'):
        try:
            img = plt.imread(path)
            test_images.append(img)
        except Exception as e:
            print(f"无法加载图像 {path}: {str(e)}")

    if not test_images:
        raise ValueError("没有找到可用的测试图像")

    weights_path = 'car1.h5'
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"模型文件 {weights_path} 不存在")

    video_name = 'parking_video.mp4'
    class_dictionary = {0: 'empty', 1: 'occupied'}

    try:
        park = Parking()
        park.show_images(test_images)
        final_spot_dict = img_process(test_images, park)
        model = keras_model(weights_path)
        img_test(test_images, final_spot_dict, model, class_dictionary, park)
        video_test(video_name, final_spot_dict, model, class_dictionary, park)
    except Exception as e:
        print(f"程序运行出错: {str(e)}")