# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from models.model_utils import (
    BasicBlock,
    Bottleneck,
    segmenthead,
    DAPPM,
    PAPPM,
    PagFM,
    Bag,
    Light_Bag,
)
import logging

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


class PIDNet(nn.Module):

	def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm_planes=96, head_planes=128, augment=True):
		super(PIDNet, self).__init__()
		self.augment = augment
		
		# I Branch
		self.conv1 =  nn.Sequential(
						  nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
						  BatchNorm2d(planes, momentum=bn_mom),
						  nn.LeakyReLU(inplace=True),
						  nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
						  BatchNorm2d(planes, momentum=bn_mom),
						  nn.LeakyReLU(inplace=True),
					  )

		self.relu = nn.LeakyReLU(inplace=True)
		self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
		self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
		self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
		self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
		self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)
		
		# P Branch
		self.compression3 = nn.Sequential(
										  nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
										  BatchNorm2d(planes * 2, momentum=bn_mom),
										  )

		self.compression4 = nn.Sequential(
										  nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
										  BatchNorm2d(planes * 2, momentum=bn_mom),
										  )
		self.pag3 = PagFM(planes * 2, planes)
		self.pag4 = PagFM(planes * 2, planes)

		self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
		self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
		self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
		
		# D Branch
		if m == 2:
			self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
			self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
			self.diff3 = nn.Sequential(
										nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
										BatchNorm2d(planes, momentum=bn_mom),
										)
			self.diff4 = nn.Sequential(
									 nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
									 BatchNorm2d(planes * 2, momentum=bn_mom),
									 )
			self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
			self.dfm = Light_Bag(planes * 4, planes * 4)
		else:
			self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
			self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
			self.diff3 = nn.Sequential(
										nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
										BatchNorm2d(planes * 2, momentum=bn_mom),
										)
			self.diff4 = nn.Sequential(
									 nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
									 BatchNorm2d(planes * 2, momentum=bn_mom),
									 )
			self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
			self.dfm = Bag(planes * 4, planes * 4)
			
		self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
		
		# Prediction Head
		if self.augment:
			self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
			self.seghead_d = segmenthead(planes * 2, planes, 1)           

		self.final_layer = segmenthead(planes * 4, head_planes, num_classes)


		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
                                
		#self.additional_conv = nn.Sequential(
            #nn.Dropout(p=0.75),
            #nn.InstanceNorm2d(256),
            #nn.Conv2d(256, 128, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #nn.InstanceNorm2d(128),
            #nn.Conv2d(128, 128, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #nn.InstanceNorm2d(128),
            #nn.Conv2d(128, 128, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
           
            #nn.InstanceNorm2d(128),
            #nn.Conv2d(128, 128, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #nn.InstanceNorm2d(128),
            #nn.Conv2d(128, 128, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #nn.InstanceNorm2d(128),
            #nn.Conv2d(128, 256, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),

            #nn.MaxPool2d(kernel_size=2, stride=2)
        #)

	def _make_layer(self, block, inplanes, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
			)

		layers = []
		layers.append(block(inplanes, planes, stride, downsample))
		inplanes = planes * block.expansion
		for i in range(1, blocks):
			if i == (blocks-1):
				layers.append(block(inplanes, planes, stride=1, no_relu=True))
			else:
				layers.append(block(inplanes, planes, stride=1, no_relu=False))

		return nn.Sequential(*layers)
	
	def _make_single_layer(self, block, inplanes, planes, stride=1):
		downsample = None
		if stride != 1 or inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
			)

		layer = block(inplanes, planes, stride, downsample, no_relu=True)
		
		return layer

	def forward(self, x):
		# width_output = torch.div(x.shape[-1], 8.0)
		# height_output = torch.div(x.shape[-2], 8.0)
		cons = 8
		width_output = int(x.shape[-1] / cons)
		height_output = int(x.shape[-2] /cons)

		x = self.conv1(x)
		x = self.layer1(x)
		x = self.relu(self.layer2(self.relu(x)))
		x_ = self.layer3_(x)
		x_d = self.layer3_d(x)
		
		x = self.relu(self.layer3(x))
		x_ = self.pag3(x_, self.compression3(x))
		x_d = x_d + F.interpolate(
				self.diff3(x),
				size=[height_output, width_output],
				mode='bilinear',
				align_corners=algc
		)

		if self.augment:
			temp_p = x_
		
		x = self.relu(self.layer4(x))
		x_ = self.layer4_(self.relu(x_))
		x_d = self.layer4_d(self.relu(x_d))
		
		x_ = self.pag4(x_, self.compression4(x))
		# print("a")
		x_d = x_d + F.interpolate(
			self.diff4(x),
			size=[height_output, width_output],
			mode='bilinear',
			align_corners=algc
		)
		# print("b")
		if self.augment:
			temp_d = x_d
			
		x_ = self.layer5_(self.relu(x_))
		x_d = self.layer5_d(self.relu(x_d))
		x = F.interpolate(
			self.spp(self.layer5(x)),
			size=[height_output, width_output],
			mode='bilinear',
			align_corners=algc
		)

        #added additional convolutional layers at the output
		#temp=self.dfm(x_, x, x_d)
		#temp2 = self.additional_conv(temp1)
		#temp3=temp1+temp2
		#x_ = self.final_layer(temp)

		x_ = self.final_layer(self.dfm(x_, x, x_d))


		if self.augment: 
			x_extra_p = self.seghead_p(temp_p)
			x_extra_d = self.seghead_d(temp_d)
			return [x_extra_p, x_, x_extra_d]
		else:
			return x_


def get_pidnet_model(
    model_size: Literal["small", "medium", "large"], num_of_classes: int
) -> PIDNet:
    if model_size == "small":
        return PIDNet(
            m=2,
            n=3,
            num_classes=num_of_classes,
            planes=32,
            ppm_planes=96,
            head_planes=128,
            augment=True,
        )
    elif model_size == "medium":
        return PIDNet(
            m=2,
            n=3,
            num_classes=num_of_classes,
            planes=64,
            ppm_planes=96,
            head_planes=128,
            augment=True,
        )
    elif model_size == "large":
        return PIDNet(
            m=3,
            n=4,
            num_classes=num_of_classes,
            planes=64,
            ppm_planes=112,
            head_planes=256,
            augment=True,
        )
    else:
        raise Exception("model_size has to be small, medium or large only.")


def get_pidnet_model_no_aug(
    model_size: Literal["small", "medium", "large"], num_of_classes: int
) -> PIDNet:
    if model_size == "small":
        return PIDNet(
            m=2,
            n=3,
            num_classes=num_of_classes,
            planes=32,
            ppm_planes=96,
            head_planes=128,
            augment=False,
        )
    elif model_size == "medium":
        return PIDNet(
            m=2,
            n=3,
            num_classes=num_of_classes,
            planes=64,
            ppm_planes=96,
            head_planes=128,
            augment=False,
        )
    elif model_size == "large":
        return PIDNet(
            m=3,
            n=4,
            num_classes=num_of_classes,
            planes=64,
            ppm_planes=112,
            head_planes=256,
            augment=False,
        )
    else:
        raise Exception("model_size has to be small, medium or large only.")


def load_pretrained_pt_file(model: PIDNet, pt_file_path_str: str):
    """
    Loads up a pretrained model into PIDNet
    """
    print("Loading pretrained weights")

    loaded_param_dict = torch.load(pt_file_path_str, map_location="cpu")
    model_param_slots = model.state_dict()
    print(f"number of pretrained params {len(loaded_param_dict)}")
    print(f"number of model params {len(model_param_slots)}")

    match_dict = {}
    # unmatch_dict = {}
    for k, v in loaded_param_dict.items():
        new_k = k.replace("model.", "")

        if new_k in model_param_slots.keys() and v.shape == model_param_slots[new_k].shape:
            match_dict[new_k] = v

            # print(f'matched: {k}:{new_k}')
        # else:
        # 	unmatch_dict[k] = v
        # print(f'unmatched: {k}:{new_k}')

    # print(len(match_dict))
    print(f"number of matching params: {len(match_dict)}")
    model_param_slots.update(match_dict)

    model.load_state_dict(model_param_slots, strict=False)

    print(f"Loaded {len(match_dict)} parameters!")

    return model

    # else:
    # 	print("using diff pretrained")
    # 	pretrained_dict = torch.load(cfg.TEST.MODEL_FILE, map_location='cpu')
    # 	if 'state_dict' in pretrained_dict:
    # 		pretrained_dict = pretrained_dict['state_dict']
    # 	model_dict = model.state_dict()
    # 	# for k, v in pretrained_dict.items():
    # 	#	print("k, v are : ", k, v.size())
    # 	#  and k!= "model.final_layer.conv2.weight" and k!= "model.final_layer.conv1.weight"
    # 	pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if
    # 					   (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    # 	msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    # 	logging.info('Attention!!!')
    # 	logging.info(msg)
    # 	logging.info('Over!!!')
    # 	# for k, v in pretrained_dict.items():
    # 	#	print("k, v are : ", k, v.size())
    # 	model_dict.update(pretrained_dict)
    # 	model.load_state_dict(model_dict, strict=False)

    # return model

from icecream import ic 

def get_seg_model(cfg, imgnet_pretrained):  # load pth.tar
    # if "s" in cfg.MODEL.NAME:
    #     model = PIDNet(
    #         m=2,
    #         n=3,
    #         num_classes=cfg.DATASET.NUM_CLASSES,
    #         planes=32,
    #         ppm_planes=96,
    #         head_planes=128,
    #         augment=True,
    #     )
    # elif "m" in cfg.MODEL.NAME:
    #     model = PIDNet(
    #         m=2,
    #         n=3,
    #         num_classes=cfg.DATASET.NUM_CLASSES,
    #         planes=64,
    #         ppm_planes=96,
    #         head_planes=128,
    #         augment=True,
    #     )
    # else:
    model = PIDNet(
        m=3,
        n=4,
        num_classes=cfg.DATASET.NUM_CLASSES,
        planes=64,
        ppm_planes=112,
        head_planes=256,
        augment=True,
    )

    if imgnet_pretrained:
        pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location="cpu") # ["state_dict"]
        model_dict = model.state_dict()
        print(f"number of pretrained params {len(pretrained_state)}")
        print(f"number of model params {len(model_dict)}")
        match_dict = {}
        for k, v in pretrained_state.items():
            new_k = k.replace("model.", "")
            if new_k in model_dict and v.shape == model_dict[new_k].shape:
                match_dict[new_k] = v
        print(f"number of matching params: {len(match_dict)}")
        model_dict.update(match_dict)
        model.load_state_dict(model_dict, strict=False)
        msg = "Loaded {} parameters!".format(len(pretrained_state))
        logging.info("Attention!!!")
        logging.info(msg)
        logging.info("Over!!!")

    else:
        print("using diff pretrained")
        print(f"Load from {cfg.TEST.MODEL_FILE}")
        pretrained_dict = torch.load(cfg.TEST.MODEL_FILE, map_location = {"cuda:0":"cpu"})
        if "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]
        # ic(pretrained_dict)
        model_dict = model.state_dict()
        # ic(model_dict.keys())
        # ic(model_dict)
        # ic(pretrained_dict.keys())
        loaded_dict = {}
        for k, v in pretrained_dict.items():
            if k.startswith('model.'):
                k = k.replace('model.', '')
            if k in model_dict.keys() and v.shape == model_dict[k].shape:
                loaded_dict[k] = v
            
            
 
        # pretrained_dict = {
        #     k[6:]: v
        #     for k, v in pretrained_dict.items()
        #     if k.starts('model.') and (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)
        # }  
		# pretrained: 'model.layer5.0.conv2.weight'
		# model_dict: 'final_layer.conv2.weight'

        msg = "Loaded {} parameters!".format(len(loaded_dict))
        logging.info("Attention!!!")
        print('len of loaded_dict', len(loaded_dict))
        logging.info(msg)
        logging.info("Over!!!")
        # for k, v in pretrained_dict.items():
        # 	print("k, v are : ", k, v.size())
        model_dict.update(loaded_dict)
        model.load_state_dict(model_dict, strict=False)

    return model


def get_pred_model(name, num_classes):
    if "s" in name:
        model = PIDNet(
            m=2,
            n=3,
            num_classes=num_classes,
            planes=32,
            ppm_planes=96,
            head_planes=128,
            augment=False,
        )
    elif "m" in name:
        model = PIDNet(
            m=2,
            n=3,
            num_classes=num_classes,
            planes=64,
            ppm_planes=96,
            head_planes=128,
            augment=False,
        )
    else:
        model = PIDNet(
            m=3,
            n=4,
            num_classes=num_classes,
            planes=64,
            ppm_planes=112,
            head_planes=256,
            augment=False,
        )

    return model


if __name__ == "__main__":
    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    device = torch.device("cuda")
    model = get_pred_model(name="pidnet_s", num_classes=19)
    model.eval()
    model.to(device)
    iterations = None

    input = torch.randn(1, 3, 1024, 2048).cuda()
    with torch.no_grad():
        output = model(input)
    print(output.shape)
    # 	for _ in range(10):
    # 		model(input)
    #
    # 	if iterations is None:
    # 		elapsed_time = 0
    # 		iterations = 100
    # 		while elapsed_time < 1:
    # 			torch.cuda.synchronize()
    # 			torch.cuda.synchronize()
    # 			t_start = time.time()
    # 			for _ in range(iterations):
    # 				model(input)
    # 			torch.cuda.synchronize()
    # 			torch.cuda.synchronize()
    # 			elapsed_time = time.time() - t_start
    # 			iterations *= 2
    # 		FPS = iterations / elapsed_time
    # 		iterations = int(FPS * 6)
    #
    # 	print('=========Speed Testing=========')
    # 	torch.cuda.synchronize()
    # 	torch.cuda.synchronize()
    # 	t_start = time.time()
    # 	for _ in range(iterations):
    # 		model(input)
    # 	torch.cuda.synchronize()
    # 	torch.cuda.synchronize()
    # 	elapsed_time = time.time() - t_start
    # 	latency = elapsed_time / iterations * 1000
    # torch.cuda.empty_cache()
    # FPS = 1000 / latency
    # print(FPS)
