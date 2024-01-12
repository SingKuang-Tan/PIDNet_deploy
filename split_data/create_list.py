import os
import glob
from pathlib import Path
import random
import collections
import cv2
import bisect
import numpy as np
import argparse
from icecream import ic
import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

main_path = osp.join(this_dir, '..')
# ic(sys.path)
add_path(main_path)



from configs import config
from configs import update_config
from configs import constants as C 

from helps import * 


def resize_img_label_from_dir(dataset_dir: str, category: list, aim_category: list, \
                              support_format: list, aim_size: int):
    count = 0
    for dd, folder, ffs in os.walk(os.path.join(dataset_dir, category[0])):
        for ff in ffs:
            if ff[-4:] in support_format:
                from_path_img = os.path.join(dd, ff)
                from_path_label = from_path_img.replace(category[0], category[1], 1)

                if not os.path.exists(from_path_label):
                    # ic(from_path_label)
                    continue

                aim_path_label = from_path_label.replace(
                    category[1], aim_category[1], 1
                )
                aim_path_img = from_path_img.replace(category[0], aim_category[0], 1)
                img = resize_img(from_path_img, aim_size)
                save_img(img, aim_path_img)

                label = resize_label(from_path_label)
                # label = map_label(label, C.MAP_FROM_MAVIS_TO_TRAIN)
                save_img(label, aim_path_label)

                count += 1
    # ic(count)
    return


def generate_list_from_dir(dataset_dir: str, category: list, support_format: list):

    imgs_labels_path = []
    nonexist_path = []
    for directory, folder, files in os.walk(dataset_dir + f"/{category[0]}/"):
        for ff in files:
        
            if ff[-4:] in support_format:
                # ic(ff)
                img_path = os.path.join(directory, ff)
                label_path = img_path.replace(category[0], category[1], 1)

                if not os.path.isfile(label_path):
                    nonexist_path.append(f'{img_path} None')
                else:

                # temp_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                # if max(np.unique(temp_label)) >= 20:
                #     print('over 20:', label_path)  # chech the label accurate
                #     nonexist_path.append(label_path)
                #     continue

                    temp = f"{img_path} {label_path}"
                    imgs_labels_path.append(temp)

    return imgs_labels_path, nonexist_path


def get_phases(imgs_labels_path: list, phases: list, indx_phases: list):
    path_phase = []
    for i in range(len(imgs_labels_path)):
        num = random.randint(1, indx_phases[-1])  # [start, stop]
        phase_indx = bisect.bisect_left(indx_phases, num)
        path_phase.append(phases[phase_indx])
    return path_phase


def save_img_label_to_text(imgs_labels_path, text_path, is_start = False):

    if not is_start:
        with open(text_path, "a+") as f:
            f.write('\n')
            f.write("\n".join(imgs_labels_path))
    else:
        with open(text_path, "w") as f:
            f.write("\n".join(imgs_labels_path))


def save_img_to_text(imgs_labels_path, text_path, is_start = False):

    if not is_start:
        with open(text_path, "a+") as f:
            f.write('\n')
            f.write('\n'.join([img_label_path.split(' ')[0] for img_label_path in imgs_labels_path]))
    else:
        with open(text_path, "w") as f:
            f.write('\n'.join([img_label_path.split(' ')[0] for img_label_path in imgs_labels_path]))



def parse_args():
	parser = argparse.ArgumentParser(description='Train segmentation network')
	
	parser.add_argument(
		'--cfg_file_path',
		help='experiment configure file name',
		default="/home/ubuntu/pidnet-linlin/configs/Mavis/20230417_pidnet_l_linlin_4gpu.yaml",
		type=str
	)
	parser.add_argument('--seed', type=int, default=304)    
	parser.add_argument('opts',
						help="Modify config options using the command-line",
						default=None,
						nargs=argparse.REMAINDER)

	args = parser.parse_args()
	cfg = update_config(config, args)

	return args

# python3 split_data/create_list.py --cfg_file_path configs/Mavis/20230503_a100.yaml
# 1) just do resize, no labelling mapping 


if __name__ == '__main__':
    args = parse_args()

    list_dir = config.DATASET.ROOT
    ic(config)
    nonqualified_name = config.DATASET.NONQUALIFIED_SET
    list_phases = [
                    config.DATASET.TRAIN_SET.split('.')[0], \
                    config.DATASET.VALID_SET.split('.')[0], \
                    config.DATASET.TEST_SET.split('.')[0]
                    ]
    
    os.makedirs(list_dir, exist_ok=True)
    category = C.FROM_CATEGORY
    aim_category = C.AIM_CATEGORY
    support_format = C.SUPPORT_FORMAT
    indx_phases = C.INDEX_PHASES
    dataset_dirs = C.BATCHES
    aim_size = C.AIM_SIZE

    count = collections.defaultdict(int)

    for i, dataset_dir in enumerate(dataset_dirs):
        ic(dataset_dir)
        if aim_category[1] not in os.listdir(dataset_dir):
            
            ic(dataset_dir, 'in resizing')
            resize_img_label_from_dir(dataset_dir, category, aim_category, support_format, aim_size)

        temp_imgs_labels_text, nonexist_labels = generate_list_from_dir(dataset_dir, aim_category, support_format)
        nonqualifieds_path = os.path.join(list_dir, nonqualified_name)
        save_img_to_text(nonexist_labels, nonqualifieds_path, is_start = i == 0)

        temp_path_phase = get_phases(temp_imgs_labels_text, list_phases, indx_phases)

        for phase in list_phases:
            text_path = os.path.join(list_dir, phase) + '.txt'
            temp = [tt for tt, pp in zip(temp_imgs_labels_text, temp_path_phase) \
                    if pp == phase]
            if len(temp) > 6000:
                temp = temp[:6000]
            ic(text_path)

            ic(len(temp))
            count[phase] += len(temp)

            if 'test' in phase:
                save_img_to_text(temp, text_path, is_start = i==0)
            else:
                save_img_label_to_text(temp, text_path, is_start = i==0)

