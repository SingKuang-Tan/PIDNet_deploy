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
from utils.function import testval, test, validate, val_miou
from utils.utils import create_logger



def parse_args():
	parser = argparse.ArgumentParser(description='Train segmentation network')
	
	parser.add_argument(
		'--cfg_file_path',
		help='experiment configure file name',
		default="/home/ubuntu/pidnet-linlin/configs/Mavis/20230417_pidnet_l_linlin_4gpu.yaml",
		type=str
	)

	parser.add_argument(
		'opts',
		help="Modify config options using the command-line",
		default=None,
		nargs=argparse.REMAINDER
	)

	args = parser.parse_args()
	update_config(config, args)

	return args


def main():
	args = parse_args()

	logger, final_output_dir, _ = create_logger(
		config, args.cfg_file_path, 'val_miou'
	)

	logger.info(pprint.pformat(args))
	logger.info(pprint.pformat(config))

	# cudnn related setting
	cudnn.benchmark = config.CUDNN.BENCHMARK
	cudnn.deterministic = config.CUDNN.DETERMINISTIC
	cudnn.enabled = config.CUDNN.ENABLED

	# build model
	model_size = config.MODEL.SIZE
	num_of_classes = config.DATASET.NUM_CLASSES
	pidnet_model = models.pidnet.get_pidnet_model(
		model_size=model_size,
		num_of_classes=num_of_classes
	)
	if config.TEST.MODEL_FILE.endswith('pt'):
		pretrained_pt_file_path_str = config.TEST.MODEL_FILE
		logger.info(f'=> loading model from {pretrained_pt_file_path_str}')

		model = models.pidnet.load_pretrained_pt_file(
			model=pidnet_model,
			pt_file_path_str=pretrained_pt_file_path_str
		)
	else:
		model = models.pidnet.get_seg_model(config, imgnet_pretrained = False)

	model = model.cuda()

	# prepare data
	test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
	test_dataset = datasets.Mavis_TC(
			root=config.DATASET.ROOT,
			list_path=config.DATASET.TEST_SET,
			num_classes=config.DATASET.NUM_CLASSES,
			class_weights= config.TRAIN.CLASS_WEIGHTS,
			ignore_label=config.TRAIN.IGNORE_LABEL,
			mean=config.TEST.MEAN,
			std = config.TEST.STD,
			do_augment = False,
			do_test = False,
	)

	testloader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=1,
		shuffle=False,
		num_workers=1,
		pin_memory=True
	) 
	# print(next(iter(testloader))[-1])
	
	start = timeit.default_timer()


	# elif ('val' in config.DATASET.TEST_SET):
	imgs_list, mious_list = val_miou(config,
                                    test_dataset,
                                    testloader,
                                    model,
                                    sv_dir=final_output_dir, 
				                    ntop = 20)


	logging.info(f'imgs_list: {imgs_list}')
	logging.info(f'mious_list: {mious_list}')


	end = timeit.default_timer()
	logger.info('Mins: %d' % int((end-start)/60))
	logger.info('Done')


if __name__ == '__main__':
	main()
