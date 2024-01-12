# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os

import cv2
import copy
import numpy as np
from PIL import Image

import torch
from .base_dataset import BaseDataset

class Mavis(BaseDataset):
	def __init__(self, 
				 root, 
				 list_path,
				 num_classes=14,
				 multi_scale=True, 
				 flip=True, 
				 ignore_label=255, 
				 base_size=2048, 
				 crop_size=(1024, 1024),
				 scale_factor=16,
				 mean=[0.485, 0.456, 0.406], 
				 std=[0.229, 0.224, 0.225],
				 bd_dilate_size=4):

		super(Mavis, self).__init__(ignore_label, base_size,
				crop_size, scale_factor, mean, std,)

		self.root = root
		self.list_path = list_path
		self.num_classes = num_classes

		self.multi_scale = multi_scale
		self.flip = flip
		
		self.img_list = [line.strip().split() for line in open(root+list_path)]

		self.files = self.read_files()
		self.label_mapping = {0: 0,
						  1: 1,
						  2: 2,
						  3: 3,
						  4: 4,
						  5: 5,
						  6:6,
						  7:7,
						  8:8,
						  9:9,
						  10:10,
						  11:11,
						  12:12,
						  13:13}
		self.class_weights =  None
		self.bd_dilate_size = bd_dilate_size
	
	def read_files(self):
		files = []
		if 'test' in self.list_path:
			for item in self.img_list:
				image_path = item
				name = os.path.splitext(os.path.basename(image_path[0]))[0]
				files.append({
					"img": image_path[0],
					"name": name,
				})
		else:
			for item in self.img_list:
				image_path, label_path = item
				name = os.path.splitext(os.path.basename(label_path))[0]
				files.append({
					"img": image_path,
					"label": label_path,
					"name": name
				})
		return files
		
	def convert_label(self, label, inverse=False):
		temp = label.copy()
		if inverse:
			for v, k in self.label_mapping.items():
				label[temp == k] = v
		else:
			for k, v in self.label_mapping.items():
				label[temp == k] = v
		return label

	def __getitem__(self, index):
		item = self.files[index]
		name = item["name"]
		image = cv2.imread(os.path.join(item["img"]), cv2.IMREAD_COLOR)
						   
		size = image.shape

		if 'test' in self.list_path:
			image = self.input_transform(image)
			image = image.transpose((2, 0, 1))

			return image.copy(), np.array(size), name

		label = cv2.imread(os.path.join(item["label"]),cv2.IMREAD_GRAYSCALE)
						   
		label = self.convert_label(label)

		image, label, edge = self.gen_sample(image, label, 
								self.multi_scale, self.flip, edge_size=self.bd_dilate_size)

		return image.copy(), label.copy(), edge.copy(), np.array(size), item["img"] # added item["img"] instead of name

	
	def single_scale_inference(self, config, model, image):
		pred = self.inference(config, model, image)
		return pred

	def convert_color(self, label, color_map):
		temp = np.zeros(label.shape + (3,)).astype(np.uint8)
		# print("labe shape: ", label.shape)
		for k,v in color_map.items():
			# print("v is: ", v, type(v))
			v.reverse()
			# print("v is: ", v, type(v))
			temp[label == k] = v
		return temp


	def save_pred(self, preds, sv_path, name, id_color_map):
		try:
			preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
			for i in range(preds.shape[0]):
				pred = self.convert_label(preds[i], inverse=True)
				save_img_id = Image.fromarray(pred)
				# save label_data
				print("name: ", name)
				print("sv_path: ", sv_path)
				if "Batch1" in name or "Batch2" in name:
					label_img_path = sv_path + "/" + name.rsplit("/", 1)[0].replace("rgb", "id").replace("/Batch1", "").replace("/Batch2", "")
				else:
					label_img_path = sv_path
				print("label_img_path: ", label_img_path)
				os.makedirs(label_img_path, exist_ok=True)
				final_id_path = label_img_path + "/" + name.rsplit("/", 1)[-1]
				if ".png" not in final_id_path:
					final_id_path += ".png"
				print("final_id_path: ", final_id_path)
				

				if "Batch1" in name or "Batch2" in name:
					color_img_path = sv_path + "/"  + name.rsplit("/", 1)[0].replace("rgb", "color").replace("/Batch1", "").replace("/Batch2", "")
				else:
					color_img_path = sv_path
				id_color_map_new = copy.deepcopy(id_color_map)
				os.makedirs(color_img_path, exist_ok=True)
				color_label = self.convert_color(pred, id_color_map_new)
				#print("color_label: ", color_label.shape, type(color_label), color_label.dtype, image_numpy.shape, type(image_numpy), image_numpy.dtype)
				save_img_color = Image.fromarray(color_label)

				final_color_path = color_img_path + "/" + name.rsplit("/", 1)[-1]
				if ".png" not in final_color_path:
					final_color_path += ".png"
				print("final_color_path: ", final_color_path)
				if final_color_path == final_id_path:
					new_final_id_path  = final_id_path.rsplit("/", 1)[0] + "/id/"
					os.makedirs(new_final_id_path, exist_ok=True)
					new_final_id_path = new_final_id_path + final_id_path.rsplit("/", 1)[-1]

					new_final_color_path  = final_color_path.rsplit("/", 1)[0] + "/color/"
					os.makedirs(new_final_color_path, exist_ok=True)
					new_final_color_path = new_final_color_path + final_color_path.rsplit("/", 1)[-1]

				save_img_id.save(new_final_id_path)
				save_img_color.save(new_final_color_path)
		except Exception as e:
			print("Exception in save_pred: ", e)
