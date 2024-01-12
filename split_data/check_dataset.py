#!/usr/bin/python

import os
import cv2
import glob
import numpy as np
import cv2

ma = 0
mi = 0
with open ('mavis/20230417_batches1_8_linlin/train.txt') as f:
	lines = f.readlines()
	for line in lines:
		files = line.replace("\n", "").split(" ")
		img1 = cv2.imread(files[0])
		img2 = cv2.imread(files[1])
		#ma = max(ma, img1.shape[0], img1.shape[1])
		#print(img1.shape, img2.shape)
		if img1.shape != img1.shape or img1.shape == (720, 1280, 3) or img2.shape == (720, 1280, 3):
			print(files[0], files[1])
		unique_items = np.unique(img2)
		print(unique_items)
		if len(unique_items) > 9 or max(unique_items) > 9:
			print("classes exceeding 9 in: ", files[1])
		break