import os
import glob
from pathlib import Path
import random
import collections
import cv2
import bisect
from icecream import ic
import numpy as np


def resize_img(file_path, aim_size=(1024, 1024)):  # aim_size should be w, h
    try:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, aim_size, interpolation=cv2.INTER_NEAREST)
        # print('write from ', file_path, 'into', aim_path)
        return img
    except:
        print(file_path, 'not existing')


def resize_label(file_path, aim_size=(1024, 1024)):  # aim_size should be w, h
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, aim_size, interpolation=cv2.INTER_NEAREST)
    # print('write from ', file_path, 'into', aim_path)
    return img


def map_label(label: np.array, mapping_dict: dict):

    new_label = np.zeros((label.shape[-2], label.shape[-1]), dtype=np.uint8)
    for k, v in mapping_dict.items():
        new_label[label == int(k)] = int(v)
    # assert sum(np.unique(label) <= 20) <= sum(np.unique(new_label) <= 20), f'label: {np.unique(label)}, new_label: {np.unique(new_label)}'

    return new_label


def save_img(img: np.array, aim_path: str):
    os.makedirs(os.path.dirname(aim_path), exist_ok=True)
    cv2.imwrite(aim_path, img)
    return
