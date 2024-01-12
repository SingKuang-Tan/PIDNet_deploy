import os 
import glob 
from pathlib import Path 
import random 
import collections 

dataset_dir = '/data/segmentation/zlin_data/pidnet_dataset/cityscapes'
category = ['rgb_1024', 'id_1024']
phase = ['train', 'valid']
list_dir = '/home/zlin/PIDNet/mavis/cityscapes_500'


dict_path = collections.defaultdict(list)
count = 0
# zurich_000121_000019_leftImg8bit.png 
# zurich_000121_000019_gtFine_labelIds.png


for ff_path in glob.glob(dataset_dir + f'/{category[0]}/*.png'):
    file_name = Path(ff_path).name
    img_path = ff_path
    label_path = ff_path.replace(category[0], category[1])
    label_path = label_path.replace('leftImg8bit.png', 'gtFine_labelIds.png')
    temp =  f'{img_path} {label_path}'  
    num = random.randint(0, 9)
    if num < 8:  # 7:3
        dict_path[phase[0]].append(temp)
    else:
        dict_path[phase[1]].append(temp)
    count += 1 
    if count >= 500:
        break

for pp in phase:
    with open(f'{list_dir}/{pp}.txt', 'w') as f:
        f.write('\n'.join(dict_path[pp]))

     

        

