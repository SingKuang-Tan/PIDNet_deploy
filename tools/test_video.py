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

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.function import *
from utils.utils import create_logger
from PIL import Image


def parse_args():
	parser = argparse.ArgumentParser(description='Train segmentation network')
	parser.add_argument('--cfg',
						help='experiment configure file name',
						default="experiments/cityscapes/pidnet_small_cityscapes.yaml",
						type=str)
	parser.add_argument('--input',
						help='input video file name',
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
	mean = [0.485, 0.456, 0.406]
	std=[0.229, 0.224, 0.225]
	image = image.astype(np.float32)[:, :, ::-1]
	image = image / 255.0
	image -= mean
	image /= std
	return image

def convert_color(label, color_map):
	print("color_map: ", color_map, type(color_map))
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

def main():
	try:
		args = parse_args()

		logger, final_output_dir, _ = create_logger(
			config, args.cfg, 'test')

		logger.info(pprint.pformat(args))
		logger.info(pprint.pformat(config))

		# cudnn related setting
		cudnn.benchmark = config.CUDNN.BENCHMARK
		cudnn.deterministic = config.CUDNN.DETERMINISTIC
		cudnn.enabled = config.CUDNN.ENABLED

		# build model
		model = models.pidnet.get_seg_model(config, imgnet_pretrained=False)
		# for name, param in model.named_parameters():
		# 	print('name: ', name)
		# 	print(type(param))
		# 	print('param.shape: ', param.shape)
		# 	print('param.requires_grad: ', param.requires_grad)
		# 	print('=====')

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

		# open color map
		id_color_map = get_color_map('configs/Mavis.yaml')
		id_color_map = reverse_color_map(id_color_map)

		# output_dir to store the predictions
		if config.TEST.MODEL_FILE:
			output_dir = config.TEST.MODEL_FILE.rsplit("/", 1)[0] + '/video_test_dir/'
			if not os.path.exists(output_dir):
				os.makedirs(output_dir)

		# write the output images to video file
		video_file_name = args.input.split("/")[-1]
		video_output_dir = 'output/Mavis/video_output/'
		os.makedirs(video_output_dir, exist_ok=True)
		Videowriter = cv2.VideoWriter(video_output_dir + video_file_name, cv2.VideoWriter_fourcc('m','p','4','v'), 10, (1920 * 2,1080))

		# open the video file
		cap = cv2.VideoCapture(args.input)
		model.eval()
		count = 0
		with torch.no_grad():
			while(cap.isOpened()):
				ret, image = cap.read()
				if ret:
					ori_height, ori_width = image.shape[0], image.shape[1]
					sv_img = np.zeros_like(image).astype(np.uint8)
					image_transformed = input_transform(image)
					image_transformed = image_transformed.transpose((2, 0, 1)).copy()
					image_tensor = torch.from_numpy(image_transformed).unsqueeze(0).cuda()
					size = image_tensor.size()

					start = time.time()
					pred_val = model(image_tensor)
					if config.MODEL.NUM_OUTPUTS > 1:
						pred_val = pred_val[config.TEST.OUTPUT_INDEX]

					# take the first list
					if pred_val.size()[-2] != ori_height or pred_val.size()[-1] != ori_width:
						pred_val = F.interpolate(input=pred_val, size=size[-2:],
							mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
						)
					#print("pred_val details after interpolation: ", pred_val.shape, type(pred_val))
					pred_val = torch.argmax(pred_val, dim=1).squeeze(0).cpu().numpy()
					#print("pred_val details after argmax: ", pred_val.shape, type(pred_val))
					for i, color in enumerate(id_color_map):
						for j in range(3):
							sv_img[:,:,j][pred_val==i] = id_color_map[i][j]

					# write it to a video
					sv_img_np = cv2.cvtColor(np.array(sv_img), cv2.COLOR_RGB2BGR)
					concat_img = cv2.hconcat([image, sv_img_np])
					Videowriter.write(concat_img)
					count += 1
				else:
					print("no frame")
					break
		print("recorded video file")
		cap.release()
		Videowriter.release()
	except Exception as e:
		print("Exception in main is: ", e)


if __name__ == '__main__':
	main()
