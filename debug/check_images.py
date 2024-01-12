item_pathes = ['/data/segmentation/Batch8/id/site0/58/frame_1240.png', \
'/data/segmentation/Batch8/id/site0/58/frame_1400.png']
# [ 1  2  7 65 66 67 68] unique_items
# [ 0  1  2  7 65 66 67 68 69] unique_items

import cv2 
import numpy as np 
for pp in item_pathes:
    temp_img = cv2.imread(pp, cv2.IMREAD_GRAYSCALE)
    unique_item = np.unique(temp_img)
    print(pp, unique_item)

# /data/segmentation/Batch8/id/site0/58/frame_1240.png [ 2  4  8 16 64 65 66 67 68]
# /data/segmentation/Batch8/id/site0/58/frame_1400.png [ 0  2  4  8 16 64 65 66 67 68 69]
