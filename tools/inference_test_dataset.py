# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
import pprint

import yaml
import logging
import time
import cv2
import glob

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch_tensorrt

from PIL import Image
import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.function import *
from utils.utils import create_logger
from utils.utils import get_confusion_matrix



def parse_args():
	parser = argparse.ArgumentParser(description='Train segmentation network')
	parser.add_argument('--cfg',
						help='experiment configure file name',
						default="experiments/cityscapes/pidnet_small_cityscapes.yaml",
						type=str)
	parser.add_argument('--test_dataset',
						help='test dataset path',
						default=None,
						type=str,
						required=True)
	parser.add_argument('opts',
						help="Modify config options using the command-line",
						default=None,
						nargs=argparse.REMAINDER)

	args = parser.parse_args()
	update_config(config, args)

	return args

def input_transform(image):
	image_tensor = input_transform_tensor(image.copy())
	mean = [0.485, 0.456, 0.406]
	std=[0.229, 0.224, 0.225]
	image = image.astype(np.float32)[:, :, ::-1]
	image = image / 255.0
	image -= mean
	image /= std
	return image, image_tensor


def input_transform_tensor(image):
	s = time.time()
	image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to("cuda")
	mean = torch.tensor([0.406, 0.456, 0.485]).to("cuda") # image type is bgr instead of rgb so flipping
	std =  torch.tensor([0.225, 0.224, 0.229]).to("cuda")
	image_tensor = torch.div(image_tensor, 255.0)
	image_tensor = torch.sub(image_tensor, mean)
	image_tensor = torch.div(image_tensor, std)
	image_tensor = torch.flip(image_tensor, [3])
	image_tensor = image_tensor.permute(0,3,1,2)
	return image_tensor

def convert_color(label, color_map):
	#print("color_map: ", color_map, type(color_map))
	try:
		temp = np.zeros(label.shape[1:] + (3,)).astype(np.uint8)
		for k,v in color_map.items():
			temp[label[0] == k] = v
		return temp
	except Exception as e:
		print("Exception in convert_color: ", e)

def get_color_map(data_cfg):
	try:
		print("Opening config file %s" % data_cfg)
		CFG = yaml.safe_load(open(data_cfg, 'r'))
	except Exception as e:
		print(e)
		print("Error opening yaml file.")
		quit()
	id_color_map = CFG["color_map"]
	return id_color_map

def reverse_color_map(id_color_map):
	for key, value in id_color_map.items():
		value.reverse()
		id_color_map[key] = value
	return id_color_map

def build_model(use_tensorrt_model):
	try:
		print("inside build model")
		cudnn.benchmark = config.CUDNN.BENCHMARK
		cudnn.deterministic = config.CUDNN.DETERMINISTIC
		cudnn.enabled = config.CUDNN.ENABLED

		# build model
		if use_tensorrt_model:
			model = models.pidnet_trt.get_seg_model(config, imgnet_pretrained=False)
		else:
			model = models.pidnet.get_seg_model(config, imgnet_pretrained=False)

		if config.TEST.MODEL_FILE:
			model_state_file = config.TEST.MODEL_FILE
		else:
			model_state_file = os.path.join(final_output_dir, 'best.pt')

		pretrained_dict = torch.load(model_state_file)
		if 'state_dict' in pretrained_dict:
			pretrained_dict = pretrained_dict['state_dict']

		model_dict = model.state_dict()
		pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
							if k[6:] in model_dict.keys()}

		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)
		model = model.cuda()
		model.eval()
		return model
	except Exception as e:
			print("Exception in build_model: ", e)

def check_trt_model(use_tensorrt_model, model_path):
	try:
		if not os.path.exists(model_path):
			net = build_model(use_tensorrt_model)

			print("ts file not found. compiling a new ts file")
			traced_model = torch.jit.trace(net, torch.empty([1, 3, 1080, 1920]).to("cuda"))
			trt_ts_module = torch_tensorrt.compile(traced_model, inputs=[torch_tensorrt.Input(
											min_shape=[1, 3, 1080, 1920],
											opt_shape=[1, 3, 1080, 1920],
											max_shape=[1, 3, 1080, 1920],
											dtype=torch.half
											)],
											require_full_compilation=True,
											enabled_precisions = {torch.half},
											truncate_long_and_double=True)
			#trt_ts_module.graph.makeMultiOutputIntoTuple()
			torch.jit.save(trt_ts_module, model_path)
			print("saved the .ts file")
		else:
			print("found trt file")
			trt_ts_module = torch.jit.load(model_path)
		return trt_ts_module
	except Exception as e:
		print("Exception in check_trt_model: ", e)

def update_confusion_matrix(label, pred_label, config, confusion_matrix):
	try:
		# print("details before confusio  matrix: ", label.shape, pred_label.shape, label.size(), config.DATASET.NUM_CLASSES,config.TRAIN.IGNORE_LABEL)
		confusion_matrix += get_confusion_matrix(
							label,
							pred_label,
							label.size(),
							config.DATASET.NUM_CLASSES,
							config.TRAIN.IGNORE_LABEL
						)
		return confusion_matrix
	except Exception as e:
		print("Exception in update_confusion_matrix: ", e)

def calculate_miou(confusion_matrix):
	try:
		#print("calculating metrics")
		pos = confusion_matrix.sum(1)
		res = confusion_matrix.sum(0)
		tp = np.diag(confusion_matrix)
		class_void = confusion_matrix.copy()
		class_void = class_void[0]
		class_void[0] = 0
		# print("tp: ", tp)
		# print("pos: ", pos)
		# print("res: ", res)
		# print("class_void: ", class_void)
		IoU_array = (tp / np.maximum(1.0, pos + res - tp - class_void))
		#print("IoU_array: ", IoU_array)
		mean_IoU = IoU_array.mean()
		#print("mean_IoU: ", mean_IoU)
		precision = tp / res
		recall = tp / pos
		#print("precision, recall: ", precision, recall)
		return  mean_IoU, IoU_array, precision, recall
	except Exception as e:
		print("Exception in calculate_miou: ", e)

def main():
	try:
		args = parse_args()
		save_pred = True
		use_tensorrt_model = True
		tensorrt_model_name = ''
		logger, final_output_dir, _ = create_logger(
			config, args.cfg, 'test')

		# open color map
		id_color_map = get_color_map('configs/Mavis.yaml')
		id_color_map = reverse_color_map(id_color_map)

		# write the output images to video file
		# test_dataset_name = args.test_dataset.split("/")[-1]
		output_dir = 'output/Mavis/test_results'
		os.makedirs(output_dir, exist_ok=True)

		count = 0
		avg_times = [0] * 3

		# get the tensorrt model
		if use_tensorrt_model:
			tensorrt_model_name = config.TEST.MODEL_FILE.split('.')[0] + '_720_1280.ts'
			print("tensorrt_model_name: ", tensorrt_model_name)
			model_trt = check_trt_model(use_tensorrt_model, tensorrt_model_name)
			print("after trt model check")
		else:
			model = build_model(use_tensorrt_model)

		size = torch.Size([1, 1, 1080, 1920])
		confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

		with torch.no_grad():
			with open(args.test_dataset) as f:
				for line in f:
					img_name = line.split(" ")[0]
					id_name = line.split(" ")[1].replace('\n', '')
					# print("img_name: ", img_name)
					mask_id = cv2.imread(id_name, cv2.IMREAD_GRAYSCALE)
					image = cv2.imread(img_name)
					ori_height, ori_width = image.shape[0], image.shape[1]

					# resize input image based on the tensorrtmodel
					if '720' in tensorrt_model_name:
						image_new = cv2.resize(image, (1280, 720))
					elif '1440' in tensorrt_model_name:
						image_new = cv2.resize(image, (1440, 1080))
					else:
						image_new = image.copy()

					# resize id image to 1080P
					#print("mask_id: ", mask_id.shape)
					if mask_id.shape != (1080,1920):
						mask_id = cv2.resize(mask_id, (1920, 1080))
					mask_id = torch.from_numpy(mask_id).long().cuda().unsqueeze(0)
					# mask_id = mask_id.permute(2,0,1)

					# preprocessing input
					start = time.time()
					image_tensor = input_transform_tensor(image_new)
					avg_times[0] = time.time() - start

					# model inference
					torch.cuda.synchronize()
					start1 = time.time()
					if use_tensorrt_model:
						pred_val = model_trt(image_tensor.half())
						pred_val= pred_val.unsqueeze(0)
						pred_val = pred_val[0]
					else:
						pred_val = model(image_tensor)
						if config.MODEL.NUM_OUTPUTS > 1:
							pred_val = pred_val[config.TEST.OUTPUT_INDEX]
					torch.cuda.synchronize()
					avg_times[1] = time.time() - start1
					# print("pred_val after predictions: ", pred_val.size(), type(pred_val))

					start = time.time()
					if pred_val.size()[-2] != ori_height or pred_val.size()[-1] != ori_width:
						# print("interpolating")
						pred_val = F.interpolate(input=pred_val, size=size[-2:],
							mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
						)
					avg_times[2] = time.time() - start
					# print("pred_val again: ", pred_val.shape, type(pred_val))

					confusion_matrix = update_confusion_matrix(mask_id, pred_val, config, confusion_matrix)
					# print("confusion_matrix: ", confusion_matrix)

					count += 1
					# if count > 20:
					# 	break
					# print("\n")

		#calculate eval metrics
		#print("confusion_matrix: ", confusion_matrix, type(confusion_matrix))
		mean_IoU, IoU_array, precision, recall = calculate_miou(confusion_matrix)
		print("mean_IoU: {}, IoU_array: {}, precision: {}, recall: {}".format(mean_IoU, IoU_array, precision, recall))
	except Exception as e:
		print("Exception in main is: ", e)


if __name__ == '__main__':
	main()