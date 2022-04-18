# Without Size consideration

import numpy as np
import glob
import cv2
from imaging_interview import preprocess_image_change_detection, compare_frames_change_detection
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
import time
import argparse
import os
import concurrent.futures


def iter_over_cameras(list_cameras, resizing_shapes):
    """ iterates the images from different camera for creating combination dictionary"""

    camera_images = list_cameras
    shapes = resizing_shapes

    combination_dict = {}
    H, W, C = shapes

    for [image_1, image_2] in tqdm(combinations(camera_images, 2)):

        if image_1 in combination_dict.keys():

            combination_dict[image_1].append(image_2)
        else:
            combination_dict[image_1] = [image_2]

    return check_for_similarity(combination_dict,H,W)



def check_for_similarity(combination_dict,H,W):
    """ Checks for similarity of the images from combination dictionary"""
    delete_list = []
    while combination_dict != {}:

        for key in tqdm(list(combination_dict)):
            if key in combination_dict.keys():
                imagespath_to_compare = combination_dict[key]
                key_image = cv2.imread(key)
                if key_image.shape != (H, W, 3):
                    key_image = cv2.resize(key_image, (H, W), interpolation=cv2.INTER_AREA)
                gray_img_1 = preprocess_image_change_detection(key_image, gaussian_blur_radius_list=[7, 7],
                                                               black_mask=(5, 10, 5, 0))

                for image_path in imagespath_to_compare:
                    if not (image_path in delete_list):
                        image = cv2.imread(image_path)
                        if image is not None:
                            if image.shape != (H, W, 3):
                                image = cv2.resize(image, (H, W), interpolation=cv2.INTER_AREA)
                            gray_img_2 = preprocess_image_change_detection(image, gaussian_blur_radius_list=[7, 7],
                                                                           black_mask=(5, 10, 5, 0))
                            result = compare_frames_change_detection(gray_img_1, gray_img_2, min_contour_area= 0.05 * H * W)[0]
                            if result < 0.1 * H * W:
                                delete_list.append(image_path)
                                if image_path in combination_dict.keys():
                                    del combination_dict[image_path]
                                    # print("deleting due to similarity : ", image_path)

                        else:
                            delete_list.append(image_path)
                            del combination_dict[image_path]
                            # print("deleting due to None error : ", image_path)
                del combination_dict[key]
    return delete_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-path', dest='input_path', type=str, help='path to data folder.', required=True)

    args = parser.parse_args()

    start_time = time.perf_counter()

    path = args.input_path #".\\dataset\\"
    images = glob.glob(path + "/*.png")
    results = []
    print(len(images))
    # Sorting the images based on its source
    camera_10 = []
    camera_20 = []
    camera_21 = []
    camera_23 = []
    delete_data = []
    delete_list = []
    for image in images:
        if "c10" in image:
            camera_10.append(image)
        if "c20" in image:
            camera_20.append(image)
        if "c21" in image:
            camera_21.append(image)
        if "c23" in image:
            camera_23.append(image)

    # print("Images in camera_10 : ", len(camera_10), "Images in camera_20 : ",  len(camera_20), "Images in camera_21 : ",len(camera_21), "Images in camera_23 : ", len(camera_23))

    list_cameras = [camera_10, camera_20, camera_21, camera_23]
    resizing_shapes = [(640, 480, 3), (1920, 1080, 3), (1100, 619, 3), (1920, 1080, 3)]
    # Using Concurrent for parallel computation
    with concurrent.futures.ThreadPoolExecutor() as executor:
        delete_data = (list(executor.map(iter_over_cameras, list_cameras, resizing_shapes)))

    end_time = time.perf_counter()

    print("Time taken : ", end_time - start_time)

    # print(delete_list)

    for data in delete_data:
        for image in data:
            os.remove(image)

    print("...")

