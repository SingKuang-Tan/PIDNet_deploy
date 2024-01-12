import os
import cv2
import copy
import numpy as np
import random
from PIL import Image

import albumentations as A
import torch

# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os
from icecream import ic
import cv2
import copy
import numpy as np
import random
from PIL import Image
import albumentations as A
import torch
from icecream import ic
from .base_dataset import BaseDataset
from .augment import *
from configs import constants as C

# import _init_paths
# from base_dataset import BaseDataset
# from augment import *
# import configs.constants as C

ic.configureOutput(includeContext=True)
# ic.disable()

def assert_label(label, max_class):  # class index is 8. so max class is 9
    unique_items = np.unique(label)
    if len(unique_items) > max_class or max(unique_items) > max_class:
        raise f"label is not mapping correctly"
    else:
        pass


class Mavis_TC(BaseDataset):
    def __init__(
        self,
        root,
        list_path,
        num_classes,
        mean,
        std,
        class_weights=[],
        ignore_label=0,
        do_augment=True,
        aug_human=False,
        do_test=False,
    ):
        super(Mavis_TC, self).__init__(mean, std)

        root = root if root[-1] == "/" else root + "/"
        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.img_list = [line.strip().split() for line in open(root + list_path)]
        self.do_augment = do_augment
        self.aug_human = aug_human
        self.do_test = do_test
        self.files = self.read_files()

        self.label_mapping = C.MAP_FROM_MAVIS_TO_TRAIN
        
        self.label_mapping_batch9 = C.MAP_FROM_MAVIS_TO_TRAIN_BATCH9

        self.pre_num_class = max(self.label_mapping.keys()) + 1
        self.cur_num_class = max(self.label_mapping.values()) + 1
        assert (
            self.cur_num_class == self.num_classes
        ), f"the mapped class from mavis {self.cur_num_class} != {self.num_classes}"

        # add augfile
        self.aug_human_path = C.AUG_HUMAN_PATH
        self.aug_human_list = os.listdir(self.aug_human_path + "/rgb")

        self.aug_vehicle_path = C.AUG_VEHICLE_PATH
        self.aug_vehicle_list = os.listdir(self.aug_vehicle_path + "/rgb")

        self.shadow_transform = shadow_transform()
        self.autumn_transform = autumn_transform()
        self.light_transform = light_transform()

        self.basic_transform = basic_transform(aim_size=C.AIM_SIZE)
        self.ignore_label = ignore_label

        if not self.do_test:
            if len(class_weights) != self.cur_num_class:
                self.class_weights = self.calculate_weights()
            else:
                self.class_weights = torch.tensor(class_weights).type(
                    "torch.FloatTensor"
                )
                print("load class weights from config: ", self.class_weights)

        # sequence
        self.sequence_list_root = C.SEQUENCE_LIST_ROOT
        self.sequence_file_list = C.SEQUENCE_FILE_LIST
        self.ldof_file_path = C.LDOF_FILE_PATH
        self.sequence_list = [int(line[:-5]) for line in open(self.sequence_file_list)]
        self.sequence_length = len(self.sequence_list)

    def read_files(self):
        files = []
        if self.do_test:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append(
                    {
                        "img": image_path[0],
                        "name": name,
                    }
                )
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({"img": image_path, "label": label_path, "name": name})
        return files

    def convert_label(self, label, inverse=False,isBatch9=False):
        temp = label.copy()
        if inverse:
            if isBatch9:
                for v, k in self.label_mapping_batch9.items():
                    label[temp == k] = v
            else:
                for v, k in self.label_mapping.items():
                    label[temp == k] = v
        else:
            if isBatch9:
                for k, v in self.label_mapping_batch9.items():
                    label[temp == k] = v
            else:
                for k, v in self.label_mapping.items():
                    label[temp == k] = v
        return label

    def get_id_color_map(self):
        colors = C.LABEL_TO_COLOR
        return colors

    def mavis_augment(self, image):
        p = random.random()
        if p < 0.3:
            transformed = self.shadow_transform(image=image)
        elif p < 0.65:
            transformed = self.autumn_transform(image=image)
        else:
            transformed = self.light_transform(image=image)

        #sk_transform
        p = random.random()
        if p<0.5:
            image[:,:,0]=image[:,:,1]
            image[:,:,2]=image[:,:,1]
            return image

        return transformed["image"].copy()

    # def gamma_trans(self, img, gamma):
    # 	gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    # 	gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    # 	image_gamma_corrected = cv2.LUT(img,gamma_table)

    # 	img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 	msk = img_grayscale>128
    # 	msk = np.array([msk, msk, msk])
    # 	msk = np.moveaxis(msk, 0, 2)
    # 	return np.where(msk, image_gamma_corrected, img)

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(item["img"]), cv2.IMREAD_COLOR)
        size = np.array(image.shape[-2:])
        if self.do_test:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            print(
                "---------------------------------------------------------------------------------",
                image.shape,
            )
            return image.copy(), size, name  # it is only used with single_inference
        try:
            label = cv2.imread(item["label"], cv2.IMREAD_GRAYSCALE)
            label = self.convert_label(label,isBatch9='Batch9' in os.path.join(item["img"]))
        except:
            print(item['label'])
        assert_label(label, self.cur_num_class)
        edge = self.get_edge(label)

        # unique_labels = np.unique(label)
        # if len(unique_labels) > 9 or np.max(unique_labels) > 9:
        # 	print("class mapping incorrect in: ", item["label"])

        new_dir_img = C.IMG_DIR_PUT_MODEL
        new_dir_label = C.LABEL_DIR_PUT_MODEL
        os.makedirs(new_dir_img, exist_ok=True)
        os.makedirs(new_dir_label, exist_ok=True)

        assert self.do_test * self.do_augment == 0

        if self.aug_human:
            image, label = self.cut_paste_augment(
                image.copy(), label.copy(), aug_prob=0.3, max_aug=2, min_aug=1
            )

        if self.do_augment:
            image = self.mavis_augment(image)
            basic = self.basic_transform(image=image, image0=label, image1=edge)
            image = basic["image"]
            label = basic["image0"]
            edge = basic["image1"]
        # save before put into model
        ##cv2.imwrite(new_dir_img + "/" + str(index) + ".png", image)

        # sv_img = np.zeros_like(image).astype(np.uint8)
        id_color_map = self.get_id_color_map()
        sv_img = self.convert_color(label, id_color_map)
        ##cv2.imwrite(new_dir_label + "/" + str(index) + ".png", sv_img)
        image, label = self.gen_sample(image, label)

        ind = index
        while ind >= self.sequence_length:
            ind -= self.sequence_length

        seq1_rgb = cv2.imread(
            self.sequence_list_root + str(self.sequence_list[ind]) + ".png"
        )
        seq2_rgb = cv2.imread(
            self.sequence_list_root + str(self.sequence_list[ind] + 1) + ".png"
        )

        # print("index:{} seq1: {}, seq2: {}".format(index, self.sequence_list[ind], self.sequence_list[ind]+1))

        # old farenback method
        # seq1_gray = cv2.cvtColor(seq1, cv2.COLOR_BGR2GRAY)
        # seq2_gray = cv2.cvtColor(seq2, cv2.COLOR_BGR2GRAY)
        # floww = cv2.calcOpticalFlowFarneback(seq1_gray, seq2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # ldof method
        with open(
            self.ldof_file_path + str(self.sequence_list[ind]) + ".npy", "rb"
        ) as f:
            x = np.load(f)
            y = np.load(f)
            f.close()

        # convert the optical flow type into python
        flow = np.stack((x, y), axis=2)
        h, w = flow.shape[:2]
        flow = -flow
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]

        seq1_rgb = self.input_transform(seq1_rgb)
        seq1_rgb = seq1_rgb.transpose((2, 0, 1))
        seq2_rgb = self.input_transform(seq2_rgb)
        seq2_rgb = seq2_rgb.transpose((2, 0, 1))
        # ic| flow.shape: (1024, 1024, 2)
        # ic| seq1_rgb.shape: (3, 1024, 1024)
        # ic| seq2_rgb.shape: (3, 1024, 1024)

        return (
            image.copy(),
            label.copy(),
            edge.copy(),
            size,
            item["img"],
            seq1_rgb,
            seq2_rgb,
            flow,
        )  # item['img']: full file path

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def convert_color(self, label, color_map):
        temp = np.zeros(label.shape + (3,)).astype(np.uint8)
        # print("labe shape: ", label.shape)
        if isinstance(color_map, dict):
            for k, v in color_map.items():
                # print("v is: ", v, type(v))

                temp[label == k] = v
        else:
            for k, v in enumerate(color_map):
                # print("v is: ", v, type(v))
                temp[label == k] = v
        return temp

    def save_pred(self, preds, sv_path, name):
        if isinstance(name, str):
            name = [name]
        try:
            preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)

            for i in range(preds.shape[0]):
                img_from = name[i]
                img_aim = os.path.join(sv_path, img_from[1:])

                # save rgb
                rgb = cv2.imread(name[i])
                os.makedirs(os.path.dirname(img_aim), exist_ok=True)
                cv2.imwrite(img_aim, rgb)

                # save pred label_data
                color_map = self.get_id_color_map()
                pred_color = self.convert_color(preds[i], color_map)
                if "1024" in img_aim:
                    pred_path = img_aim.replace("rgb_1024", "pred_1024", 1)
                else:
                    pred_path = img_aim.replace("rgb", "pred", 1)
                os.makedirs(os.path.dirname(pred_path), exist_ok=True)
                cv2.imwrite(pred_path, pred_color)

                # save id data
                if "1024" in img_aim:
                    gt = cv2.imread(
                        img_from.replace("rgb_1024", "id_1024", 1), cv2.IMREAD_GRAYSCALE
                    )
                    gt_path = img_aim.replace("rgb_1024", "id_1024", 1)
                else:
                    gt = cv2.imread(
                        img_from.replace("rgb", "id", 1), cv2.IMREAD_GRAYSCALE
                    )
                    gt_path = img_aim.replace("rgb", "id", 1)
                converted_gt = self.convert_label(gt,isBatch9='Batch9' in img_from)
                gt_color = self.convert_color(converted_gt, color_map)

                os.makedirs(os.path.dirname(gt_path), exist_ok=True)
                cv2.imwrite(gt_path, gt_color)

        except Exception as e:
            print("Exception in save_pred: ", e)

    def calculate_weights(self):
        count = np.ones((self.cur_num_class))  # data labelling has 20 types
        for item in self.img_list:
            if len(item) <= 1:
                raise f"lack of rgb or labelling path for {item}"
            label_path = item[-1]
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label = self.convert_label(label,isBatch9='Batch9' in label_path)

            assert_label(label, self.cur_num_class)

            for i in range(self.cur_num_class):
                count[i] += np.sum(label == i)

        ic("number of pixels in cur classes: ", count)
        count = np.log(count)
        count = np.clip(count, 1, None)
        cw = count / np.sum(count)
        ic("After log normalization: ", count)

        cw = np.array([1 / cw[i] if cw[i] > 0 else 0 for i in range(len(cw))])
        class_weights = torch.from_numpy(cw / np.sum(cw)).type("torch.FloatTensor")

        # if 'mavis' in self.list_path:
        #     class_weights = np.array([0.0842, 0.0881, 0.0773, 0.0832, 0.0949, 0.1011, 0.0959, 0.0850, 0.1016, 0.0937, 0.0951])
        # elif 'freiburg' in self.list_path:
        #     class_weights = np.array([0.1382, 0.0100, 0.0074, 0.1382, 0.1382, 0.1382, 0.0074, 0.1382, 0.0076, 0.1382, 0.1382])
        
        print(f'for {self.list_path}')
        print("original_class_weights", class_weights)
        factor = torch.ones(self.cur_num_class)
        factor[1] = 1.3  # obstacle
        factor[3] = 1.3  # vine_canopy
        factor[4] = 1.5  # vine stem
        factor[5] = 1.4  # vine pole
        factor[6] = 1.2  # vegetation
        # factor[7] = 0.9 # hood
        # factor[8] = 2.5 # sky
        ####factor[9] = 0.5 # human
        ####factor[10] = 0.5  # vehicle
        class_weights = class_weights * factor
        class_weights = torch.tensor(class_weights / torch.sum(class_weights)).type("torch.FloatTensor")
        # print("factor: ", factor)
        print("adjusted_class_weights: ", class_weights)

        return class_weights

    def cut_paste_augment(self, image_h, label_h, aug_prob=1, min_aug=1, max_aug=3):
        if random.random() <= aug_prob:
            num_times = random.randint(min_aug, max_aug + 1)
            for i in range(num_times):
                if random.random() <= 0.6:
                    self.aug_file_list = self.aug_human_list
                    self.aug_file_path = self.aug_human_path
                    aim_class = C.HUMAN_CLASS
                else:
                    self.aug_file_list = self.aug_vehicle_list
                    self.aug_file_path = self.aug_vehicle_path
                    aim_class = C.VEHICLE_CLASS

                file_name = random.choice(self.aug_file_list)
                to_aug_rgb = cv2.imread(
                    self.aug_file_path + "/rgb/" + file_name, cv2.IMREAD_COLOR
                )

                 
                to_aug_rgb = A.RandomBrightnessContrast(brightness_limit= (-0.63, -0.23), p = 0.8)(image = to_aug_rgb)['image']
                # print('tanned the figure!')

                to_aug_id = cv2.imread(
                    self.aug_file_path + "/id/" + file_name, cv2.IMREAD_GRAYSCALE
                )  # 0, human = 1
                # ic(to_aug_id.shape)

                # if "2023" in file_name:  # resize implements
                # 	aim_time = random.randint(1, 3)
                # 	aim_size = (int(aim_time * to_aug_id.shape[1]), int(aim_time * to_aug_id.shape[0])) # h, w, r
                # 	to_aug_rgb = cv2.resize(to_aug_rgb, aim_size, interpolation = cv2.INTER_AREA)
                # 	to_aug_id = cv2.resize(to_aug_id, aim_size, interpolation = cv2.INTER_AREA)

                # ic(np.unique(to_aug_id))
                img_h, img_w, _ = image_h.shape
                aug_h, aug_w, _ = to_aug_rgb.shape

                if img_h < aug_h or img_w < aug_w:
                    to_aug_rgb = cv2.resize(to_aug_rgb, (540, 540))
                    to_aug_id = cv2.resize(to_aug_id, (540, 540))
                img_h, img_w, _ = image_h.shape
                aug_h, aug_w, _ = to_aug_rgb.shape

                try:
                    start_x = random.randrange(
                        0, img_h - aug_h
                    )  # (img_h//2 - aug_h//2)
                    start_y = random.randrange(
                        0, img_w - aug_w
                    )  # (img_w//2 - aug_w//2)
                except:
                    ic(img_h, aug_h)
                    ic(img_w, aug_w)
                msk = to_aug_id != 0

                # try:

                label_h[
                    start_x : start_x + aug_h, start_y : start_y + aug_w
                ] = np.where(
                    msk,
                    aim_class,
                    label_h[start_x : start_x + aug_h, start_y : start_y + aug_w],
                )

                msk = np.array([msk, msk, msk])
                # ic(msk.shape)
                msk = np.moveaxis(msk, 0, 2)

                # ic(start_x, start_x + aug_h, start_y, start_y + aug_w)

                image_h[
                    start_x : start_x + aug_h, start_y : start_y + aug_w, :
                ] = np.where(
                    msk,
                    to_aug_rgb,
                    image_h[start_x : start_x + aug_h, start_y : start_y + aug_w, :],
                )
                # except:
                # 	ic(msk.shape)
                # 	ic(label_h.shape)
                # 	ic(self.aug_file_path+"/id/"+file_name)
                # 	raise 'Error in copy_paste in dataset'

        return image_h, label_h


if __name__ == "__main__":
    temp_dataset = Mavis_TC(
        root="/home/zlin/PIDNet/mavis/20230323_batches1_8_linlin",
        list_path="valid.txt",
        num_classes=11,
        class_weights=[1 / 11 for i in range(11)],
        mean=(0.4374, 0.4590, 0.4385),
        std=(0.1952, 0.2018, 0.2026),
        ignore_label=0,
        do_augment=True,
        aug_human=True,
        do_test=False,
    )

    for i in range(len(temp_dataset)):
        image, label, _, _, _, _, _, _ = temp_dataset[i]
