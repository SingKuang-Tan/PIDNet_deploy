# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
import pprint

import logging
import timeit
import yaml

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.function import testval, test
from utils.utils import create_logger

def parse_args():
	parser = argparse.ArgumentParser(description='Train segmentation network')
	
	parser.add_argument('--cfg',
						help='experiment configure file name',
						default="experiments/Mavis/pidnet_small_cityscapes.yaml",
						type=str)
	parser.add_argument('opts',
						help="Modify config options using the command-line",
						default=None,
						nargs=argparse.REMAINDER)

	args = parser.parse_args()
	update_config(config, args)

	return args

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

def main():
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
	pretrained_model_name = config.TEST.MODEL_FILE.split("/")[-1]
	imgnet = 'imagenet_' in pretrained_model_name
	model = model = models.pidnet.get_seg_model(config, imgnet_pretrained=imgnet)

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

	# prepare data
	test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
	test_dataset = eval('datasets.'+config.DATASET.DATASET)(
						root=config.DATASET.ROOT,
						list_path=config.DATASET.TEST_SET,
						num_classes=config.DATASET.NUM_CLASSES,
						multi_scale=False,
						flip=False,
						ignore_label=config.TRAIN.IGNORE_LABEL,
						base_size=config.TEST.BASE_SIZE,
						crop_size=test_size)

	testloader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=1,
		shuffle=False,
		num_workers=0,
		pin_memory=True)
	
	start = timeit.default_timer()

	# get color map
	id_color_map = get_color_map('configs/Mavis.yaml')
	
	if ('test' in config.DATASET.TEST_SET):
		test(config, 
			 test_dataset, 
			 testloader, 
			 model,
			 id_color_map,
			 sv_dir=final_output_dir)
		
	elif ('val' in config.DATASET.TEST_SET):
		mean_IoU, IoU_array, pixel_acc, mean_acc = testval(config, 
														   test_dataset, 
														   testloader, 
														   model, id_color_map,
														   sv_dir=final_output_dir)
	
		msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
			Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, 
			pixel_acc, mean_acc)
		logging.info(msg)
		logging.info(IoU_array)


	end = timeit.default_timer()
	logger.info('Mins: %d' % np.int((end-start)/60))
	logger.info('Done')


if __name__ == '__main__':
	main()
