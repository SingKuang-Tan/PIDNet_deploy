#!/usr/bin/python

import os
import cv2
import glob
import numpy as np
from PIL import Image


def main():
	class_dict = {(0,0,0): 0, (246, 4, 228):1, (173, 94, 48):2, (68, 171, 117):3, (162, 122, 174):4, (121, 119, 148):5, (253, 75, 40): 6, (170, 60, 100):7, (60, 100, 179):8, (170, 100, 60):9}
	i = 0
	try:
		for file in glob.glob("/home/cv/PIDNet/all_dataset/MAVIS-3D-GROUPED/gt_color/**/*.png", recursive=True):
			image_bgr = cv2.imread(file)
			if image_bgr.shape == (720, 1280, 3):
				print("image name: ", file)
				i+=1
				continue
			new_image = np.zeros((image_bgr.shape[0],image_bgr.shape[1],3)).astype('int')
			for key, value in class_dict.items():
				new_image[(image_bgr[:,:,2]==key[2]) & (image_bgr[:,:,1]==key[1]) & (image_bgr[:,:,0]==key[0])] = np.array([value,value,value]).reshape(1,3)

			new_image = new_image[:,:,0]
			max_ = np.max(new_image)
			min_ = np.min(new_image)
			if max_ == 255 or min_ == -1:
				print("image name: ", file, min_, max_)
			output_filename = file.replace("gt_color", "id")
			# print("output_filename: ", output_filename, output_filename.rsplit("/", 1)[0])
			os.makedirs(output_filename.rsplit("/", 1)[0], exist_ok=True)
			cv2.imwrite(output_filename, new_image)
			i += 1
	except Exception as e:
		print("Exception: ", e)

if __name__ == '__main__':
	main()