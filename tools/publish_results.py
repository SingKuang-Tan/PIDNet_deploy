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

# import ros libraries
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class Segmentation:
	def __init__(self, args):
		self.args = args
		self.image_timer = None
		self.input_img = None
		self.debug = True
		self.use_tensorrt_model = True
		self.numpy_testing = False
		update_config(config, args)
		self.bridge = CvBridge()
		rospy.init_node("segmentation", anonymous=False)
		self.loop_rate = rospy.Rate(10)
		self.initialize_vars()
		rospy.Subscriber('/zed_front/image', Image, self.image_callback)
		self.image_pub = rospy.Publisher('segmentation/image', Image, queue_size=10)
		self.debug_image_pub = rospy.Publisher('segmentation/image_debug', Image, queue_size=10)
		self.confidence_image_pub = rospy.Publisher('segmentation/confidence_values', Image, queue_size=10)
		self.main()

	def initialize_vars(self):
		classes = config.DATASET.NUM_CLASSES
		colors = torch.tensor([[0, 0, 0], [246, 4, 228], [173, 94, 48], [68, 171, 117], [162, 122, 174], [121, 119, 148], [253, 75, 40], [170, 60, 100], [60, 100, 179], [170, 100, 60]]).cuda()
		self.remapping = torch.arange(0, classes).cuda(), colors

	# callback function to read the image data
	def image_callback(self, msg):
		try:
			# Convert your ROS Image message to OpenCV2
			self.image_timer = time.time()
			self.input_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
			#print("image shape: ", self.input_img.shape)
		except Exception as e:
			print("Exception in image callback is: ",  e)

	def input_transform(self, image):
		image_tensor = input_transform_tensor(image.copy())
		mean = [0.485, 0.456, 0.406]
		std=[0.229, 0.224, 0.225]
		image = image.astype(np.float32)[:, :, ::-1]
		image = image / 255.0
		image -= mean
		image /= std
		return image, image_tensor


	def input_transform_tensor(self, image):
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

	def convert_color(self, label, color_map):
		#print("color_map: ", color_map, type(color_map))
		try:
			temp = np.zeros(label.shape[1:] + (3,)).astype(np.uint8)
			for k,v in color_map.items():
				temp[label[0] == k] = v
			return temp
		except Exception as e:
			print("Exception in convert_color: ", e)

	def get_color_map(self, data_cfg):
		try:
			print("Opening config file %s" % data_cfg)
			CFG = yaml.safe_load(open(data_cfg, 'r'))
		except Exception as e:
			print(e)
			print("Error opening yaml file.")
			quit()
		id_color_map = CFG["color_map"]
		return id_color_map

	def reverse_color_map(self, id_color_map):
		for key, value in id_color_map.items():
			value.reverse()
			id_color_map[key] = value
		return id_color_map

	def build_model(self):
		try:
			# cudnn related setting
			print("inside build model")
			cudnn.benchmark = config.CUDNN.BENCHMARK
			cudnn.deterministic = config.CUDNN.DETERMINISTIC
			cudnn.enabled = config.CUDNN.ENABLED

			# build model
			if self.use_tensorrt_model:
				self.model = models.pidnet_trt.get_seg_model(config, imgnet_pretrained=False)
			else:
				self.model = models.pidnet.get_seg_model(config, imgnet_pretrained=False)

			if config.TEST.MODEL_FILE:
				model_state_file = config.TEST.MODEL_FILE
			else:
				model_state_file = os.path.join(final_output_dir, 'best.pt')

			pretrained_dict = torch.load(model_state_file)
			if 'state_dict' in pretrained_dict:
				pretrained_dict = pretrained_dict['state_dict']
			#print("model: ", self.model)
			model_dict = self.model.state_dict()
			pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
								if k[6:] in model_dict.keys()}

			model_dict.update(pretrained_dict)
			self.model.load_state_dict(model_dict)
			self.model = self.model.cuda()
			self.model.eval()
		except Exception as e:
				print("Exception in build_model: ", e)

	def check_trt_model(self, model_path):
		try:
			if not os.path.exists(model_path):
				print("ts file not found. compiling a new ts file")

				# build the model and load the config file
				self.build_model()

				# create tensorrt module
				traced_model = torch.jit.trace(self.model, torch.empty([1, 3, 1080, 1920]).to("cuda"))
				self.trt_ts_module = torch_tensorrt.compile(traced_model, inputs=[torch_tensorrt.Input(
								min_shape=[1, 3, 1080, 1920],
								opt_shape=[1, 3, 1080, 1920],
								max_shape=[1, 3, 1080, 1920],
								dtype=torch.half
								)],
								require_full_compilation=True,
								enabled_precisions = {torch.half},
								truncate_long_and_double=True)
				torch.jit.save(self.trt_ts_module, model_path)
				print("saved the tensorrt model")
			else:
				print("loading saved tensorrt model file")
				self.trt_ts_module = torch.jit.load(model_path)
		except Exception as e:
				print("Exception in check_trt_model: ", e)


	def post_process(self, tensor):
		s= time.time()
		index = torch.bucketize(tensor.ravel(), self.remapping[0])
		s1 = time.time()
		tensor_color = self.remapping[1][index].reshape(torch.Size([1080, 1920, 3]))
		return tensor_color.to(torch.uint8)

	def to_numpy(self, tensor):
		#print("tensor.requires_grad: ", tensor.requires_grad, tensor.shape)
		return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

	def main(self):
		try:
			# open color map
			id_color_map = self.get_color_map('configs/Mavis.yaml')
			id_color_map = self.reverse_color_map(id_color_map)
			avg_times = [0] * 8

			# get the tensorrt model
			if self.use_tensorrt_model:
				tensorrt_model_name = config.TEST.MODEL_FILE.split('.')[0] + '.ts'
				print("tensorrt_model_name: ", tensorrt_model_name)
				model_trt = self.check_trt_model(tensorrt_model_name)
				#print("after trt model check")
			else:
				self.build_model()

			with torch.no_grad():
				while not rospy.is_shutdown():
					if(self.input_img is not None) and (self.image_timer is not None) and (time.time() - self.image_timer < 0.5):
						image = self.input_img.copy()
						ori_height, ori_width = image.shape[0], image.shape[1]
						sv_img = np.zeros_like(image).astype(np.uint8)

						start = time.time()
						if self.numpy_testing:
							image_transformed, image_tensor_first = self.input_transform(image)
							print("time for image_transformed: ",  time.time() - start)
							start1 = time.time()
							image_transformed = image_transformed.transpose((2, 0, 1)).copy()
							image_tensor = torch.from_numpy(image_transformed).unsqueeze(0).to("cuda")
							image_tensor = image_tensor.contiguous()
						else:
							image_tensor = self.input_transform_tensor(image)
						avg_times[0] = time.time() - start

						size = image_tensor.size()
						torch.cuda.synchronize()
						start1 = time.time()
						if self.use_tensorrt_model:
							pred_val = self.trt_ts_module(image_tensor.half())
							pred_val= pred_val.unsqueeze(0)
							pred_val = pred_val[0]
						else:
							pred_val = self.model(image_tensor)
							if config.MODEL.NUM_OUTPUTS > 1:
								pred_val = pred_val[config.TEST.OUTPUT_INDEX]
						torch.cuda.synchronize()
						avg_times[1] = time.time() - start1

						start1 = time.time()
						if pred_val.size()[-2] != ori_height or pred_val.size()[-1] != ori_width:
							pred_val = F.interpolate(input=pred_val, size=size[-2:],
								mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
							)
						avg_times[2] = time.time() - start1

						# calculate softmax of the output vals
						start1 = time.time()
						sm = torch.nn.Softmax(dim=1)
						prob_predictions = sm(pred_val)
						#print("prob_predictions: ", prob_predictions.shape, prob_predictions[0][0][0])
						probs_val = torch.max(prob_predictions, dim=1).values.squeeze(0)
						probs_val = torch.mul(probs_val, 100).to(torch.uint8).cpu().numpy()
						#print("probs_val: ", probs_val.shape, probs_val[0][0])

						pred_val = torch.argmax(pred_val, dim=1).squeeze(0).to(torch.uint8)
						pred_val_numpy = pred_val.cpu().numpy()
						# print("pred_val_numpy: ", pred_val_numpy.shape, pred_val_numpy.dtype)
						#print("time for pred_val conversion: ", time.time() - start1)
						avg_times[3] = time.time() - start1

						start1 = time.time()
						if self.numpy_testing:
							for i, color in enumerate(id_color_map):
								for j in range(3):
									sv_img[:,:,j][pred_val_numpy==i] = id_color_map[i][j]
							color_img_numpy = cv2.cvtColor(np.array(sv_img), cv2.COLOR_RGB2BGR)

						else:
							s1 = time.time()
							color_img_tensor = self.post_process(pred_val)
							s1 = time.time()
							color_img_numpy = self.to_numpy(color_img_tensor)
							#print("time for color_image numpy conversion: ", time.time() - s1, color_img_tensor.shape, color_img_tensor.dtype)
						avg_times[4] = time.time() - start1

						# publish the id mask
						start1 = time.time()
						msg = self.bridge.cv2_to_imgmsg(pred_val_numpy, "mono8")
						self.image_pub.publish(msg)
						avg_times[5] = time.time() - start1

						# publish the confidence values
						start1 = time.time()
						msg = self.bridge.cv2_to_imgmsg(probs_val, "mono8")
						self.confidence_image_pub.publish(msg)
						avg_times[6] = time.time() - start1

						start1 = time.time()
						# concatenate input image and color output
						#print("image, color_img_numpy: ", image.shape, color_img_numpy.shape, image.dtype, color_img_numpy.dtype)
						img_combined = cv2.hconcat([image, color_img_numpy])

						# publish the concatenated image
						msg = self.bridge.cv2_to_imgmsg(img_combined, "bgr8") #encoding="passthrough")
						self.debug_image_pub.publish(msg)
						avg_times[7] = time.time() - start1

						# print the average times for debugging
						print("avg_times: ", avg_times)
						print("Total time: ", time.time() - start)
						print("\n")
					else:
						print("no frame received")
					self.loop_rate.sleep()
		except Exception as e:
			print("Exception in main is: ", e)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train segmentation network')
	parser.add_argument('--cfg',
						help='experiment configure file name',
						default="configs/Mavis/pidnet_medium.yaml",
						type=str)
	parser.add_argument('opts',
						help="Modify config options using the command-line",
						default=None,
						nargs=argparse.REMAINDER)
	args = parser.parse_args()
	seg_obj = Segmentation(args)
