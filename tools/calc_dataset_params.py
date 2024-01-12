import torch
from torch.utils.data import DataLoader
import cv2

class bd(torch.utils.data.Dataset):
    def __init__(self, root=None, list_path=None, ontology=None):
        #super(self).__init__(self)
        self.list_path = list_path
        self.root = root
        self.img_list = [line.strip().split() for line in open(list_path)]
        self.label_mapping = {
                    0: 0,
                    1: 1,
                    2: 1,
                    3: 2,
                    4: 2,
                    5: 3,
                    6: 2,
                    7: 4,
                    8: 2,
                    9: 5,
                    10: 2,
                    11: 6,
                    12: 6,
                    13: 2,
                    14: 1,
                    15: 1,
                    16: 7,
                    17: 8,
                    18: 9
                }
        #print(self.img_list)
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, index):
        rgb = cv2.imread(self.img_list[index][0])
        label = cv2.imread(self.img_list[index][1], cv2.IMREAD_GRAYSCALE)
        rgb = rgb/255
        rgb = rgb.transpose((2, 0, 1))
        label = self.convert_label(label)
        return rgb, label

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label
        


def batch_mean_and_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    print("Mean:", mean)
    print("Standard deviation:", std)

def find_weights(loader, num_class):
    weights = torch.zeros(num_class)
    for _, label in loader:
        label = torch.squeeze(label)
        for i in range(num_class):
            weights[i] = weights[i]+torch.sum(label==i)
    
    print("Weights sum:", weights)
    weights = torch.log(weights)
    weights = weights/torch.sum(weights)
    print("Log normalized weights:", weights)

ds = bd(None, "/home/ubuntu/pidnet_configs/2022Oct6-anurag/train.txt")
from torch.utils.data import DataLoader

batch_size = 6
num_class = 10
loader = DataLoader(
  ds, 
  batch_size = batch_size, 
  num_workers=1)
batch_mean_and_sd(loader)

loader = DataLoader(
  ds, 
  batch_size = 1, 
  num_workers=1)
find_weights(loader, num_class)
