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
import torch_tensorrt

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
	image_tensor = input_transform_tensor(image.copy())
	mean = [0.485, 0.456, 0.406]
	std=[0.229, 0.224, 0.225]
	image = image.astype(np.float32)[:, :, ::-1]
	#print("image details: ", image.dtype, image[0][0])
	image = image / 255.0
	#print("image after div: ", image[0][0])
	image -= mean
	image /= std
	#print("image after last div: ", image[0][0])
	return image, image_tensor


def input_transform_tensor(image):
	s = time.time()
	#image_tensor = torch.from_numpy(image).unsqueeze(0).to("cuda")
	image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to("cuda")
	#print("image_tensor: ", image_tensor.dtype, image_tensor.shape, image_tensor[0][0][0])
	mean = torch.tensor([0.406, 0.456, 0.485]).to("cuda") # image type is bgr instead of rgb so flipping
	std =  torch.tensor([0.225, 0.224, 0.229]).to("cuda")
	image_tensor = torch.div(image_tensor, 255.0)
	#print("image_tensor after first div: ", image_tensor.dtype, image_tensor[0][0][0])
	image_tensor = torch.sub(image_tensor, mean)
	#print("image_tensor after sub: ", image_tensor.dtype, image_tensor[0][0][0])
	image_tensor = torch.div(image_tensor, std)
	image_tensor = torch.flip(image_tensor, [3])
	#print("image_tensor after last div and pixel flip: ", image_tensor.dtype, image_tensor[0][0][0])
	image_tensor = image_tensor.permute(0,3,1,2)
	#print("image_tensor after permute: ", image_tensor.dtype, image_tensor.shape, image_tensor[0][0][0])
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

def check_trt_model(model_path, net):
	try:
		if not os.path.exists(model_path):
			print("ts file not found. compiling a new ts file")
			traced_model = torch.jit.trace(net, torch.empty([1, 3, 1080, 1920]).to("cuda"))
			#scripted_model = torch.jit.script(net, torch.empty([1, 3, 1080, 1920]).to("cuda"))
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

def main():
	try:
		args = parse_args()
		use_tensorrt_model = True
		numpy_testing = False
		logger, final_output_dir, _ = create_logger(
			config, args.cfg, 'test')

		logger.info(pprint.pformat(args))
		logger.info(pprint.pformat(config))

		# cudnn related setting
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

		logger.info('=> loading model from {}'.format(model_state_file))
		pretrained_dict = torch.load(model_state_file)

		if 'state_dict' in pretrained_dict:
			pretrained_dict = pretrained_dict['state_dict']
		model_dict = model.state_dict()
		pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
							if k[6:] in model_dict.keys()}
		for k, _ in pretrained_dict.items():
			logger.info(
				'=> loading {} from pretrained model'.format(k))
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)
		model = model.cuda()
		model.eval()

		# open color map
		id_color_map = get_color_map('configs/Mavis.yaml')
		id_color_map = reverse_color_map(id_color_map)

		# write the output images to video file
		video_file_name = args.input.split("/")[-1]
		output_dir = 'output/Mavis/bagfiles_prediction'
		os.makedirs(output_dir, exist_ok=True)
		Videowriter = cv2.VideoWriter(output_dir + "/test.mp4", cv2.VideoWriter_fourcc('m','p','4','v'), 10, (1920 * 2,1080))

		# open the video file
		cap = cv2.VideoCapture(args.input)
		count = 0
		avg_times = [0,0,0]

		# get the tensorrt model
		if use_tensorrt_model:
			tensorrt_model_name = config.TEST.MODEL_FILE.split('.')[0] + '.ts'
			print("tensorrt_model_name: ", tensorrt_model_name)
			model_trt = check_trt_model(tensorrt_model_name, model)

		with torch.no_grad():
			while(cap.isOpened()):
				ret, image = cap.read()
				image_new = image.copy() #cv2.resize(image, (960, 540))
				if ret:
					ori_height, ori_width = image.shape[0], image.shape[1]
					sv_img = np.zeros_like(image).astype(np.uint8)
					start = time.time()
					if numpy_testing:
						image_transformed, image_tensor_first = input_transform(image)
						start1 = time.time()
						image_transformed = image_transformed.transpose((2, 0, 1)).copy()
						#print("image_transformed after transpose: ", image_transformed.dtype, image_transformed[0][0])
						image_tensor = torch.from_numpy(image_transformed).unsqueeze(0).to("cuda")
						image_tensor = image_tensor.contiguous()
					else:
						image_tensor = input_transform_tensor(image_new)

					avg_times[0] = time.time() - start
					size = image_tensor.size()
					if use_tensorrt_model:
						start = time.time()
						pred_val = model_trt(image_tensor.half())
						pred_val= pred_val.unsqueeze(0)
						avg_times[1] = time.time() - start
						pred_val = pred_val[0]
					else:
						start = time.time()
						pred_val = model(image_tensor)
						if config.MODEL.NUM_OUTPUTS > 1:
							pred_val = pred_val[config.TEST.OUTPUT_INDEX]
						avg_times[1] = time.time() - start

					if  pred_val.size()[2] == 68 and pred_val.size()[3] == 120:
						size = torch.Size([1, 1, 1080, 1920])
					print("pred_val after predictions: ", pred_val.size(), type(pred_val), size[-2:])

					# take the first list
					#pred_val = pred_val[0]
					start = time.time()
					if pred_val.size()[-2] != ori_height or pred_val.size()[-1] != ori_width:
						print("interpolating")
						pred_val = F.interpolate(input=pred_val, size=size[-2:],
							mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
						)
					avg_times[2] = time.time() - start
					#print("pred_val again: ", pred_val.shape, type(pred_val))
					pred_val = torch.argmax(pred_val, dim=1).squeeze(0).cpu().numpy()
					for i, color in enumerate(id_color_map):
						for j in range(3):
							sv_img[:,:,j][pred_val==i] = id_color_map[i][j]

					# write it to a video
					sv_img_np = cv2.cvtColor(np.array(sv_img), cv2.COLOR_RGB2BGR)
					concat_img = cv2.hconcat([image, sv_img_np])
					Videowriter.write(concat_img)
					count += 1
					if count > 500:
						break
				else:
					print("no frame")
					break
				print("avg_times: ", avg_times)
				print("\n")
		print("recorded video file")
		cap.release()
		Videowriter.release()
	except Exception as e:
		print("Exception in main is: ", e)


if __name__ == '__main__':
	main()
