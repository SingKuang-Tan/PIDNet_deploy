import PIDNet.tools._init_paths 
import torch
import cv2
import numpy as np
import models
import onnx

import onnxruntime as ort

import matplotlib.pyplot as plt

from configs import constants as C

#import pycuda.driver as cuda
#import pycuda.autoinit
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger()

def convert_color(label, color_map):
    temp = np.zeros(label.shape + (3,)).astype(np.uint8)
    # print("labe shape: ", label.shape)
    if isinstance(color_map, dict):
        for k, v in color_map.items():
            # print("v is: ", v, type(v))

            temp[label == k] = v
    else:
        for k, v in enumerate(color_map):
            # print("v is: ", v, type(v))
            temp[label == k] = v
    return temp
 
def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    ##builder.max_workspace_size = 1 << 30
    # we have only one image in batch
    builder.max_batch_size = 1
    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        #builder.fp16_mode = True
        config.set_flag(trt.BuilderFlag.FP16)

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")
 
    return engine, context

def input_transform(image):
	mean = [0.485, 0.456, 0.406]
	std=[0.229, 0.224, 0.225]
	image = image.astype(np.float32)[:, :, ::-1]
	image = image / 255.0
	image -= mean
	image /= std
	return image

input_image_file_path='/home/stan/pidnet/PIDNet/test_images/sky_images/frame_250.png'
ori_image = cv2.imread(input_image_file_path)

transformed_image = input_transform(ori_image)
transformed_image = transformed_image.transpose((2, 0, 1)).copy()
input = torch.from_numpy(transformed_image).unsqueeze(0).cuda()

model_size = 'large'
num_of_classes = 10
pidnet_model = models.pidnet2.get_pidnet_model(
    model_size=model_size,
    num_of_classes=num_of_classes
)

pretrained_pt_file_path_str = './trained_models/20231214/best.pt'

model = models.pidnet2.load_pretrained_pt_file(
        model=pidnet_model,
        pt_file_path_str=pretrained_pt_file_path_str
    )
#model.augment = False

model.eval().to("cuda") #.cuda()

#pt_path='./trained_models/20231108/best.pth.tar'
#pt_path='./PIDNet/20231018_best.pth'
#model = torch.load(pt_path)#['model_state_dict']

ONNX_FILE_PATH = 'best.onnx'
torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=['input'],
                  output_names=['output'], export_params=True)

onnx_model = onnx.load(ONNX_FILE_PATH)
onnx.checker.check_model(onnx_model)

ort_sess = ort.InferenceSession(ONNX_FILE_PATH)
outputs = ort_sess.run(None, {'input': np.expand_dims(transformed_image,axis=0)})
#print(np.shape(outputs))
outputs=outputs[0]

outputs = np.argmax(outputs, axis=1).squeeze()

outputs = np.array(outputs.squeeze())

outputs = convert_color(outputs, C.LABEL_TO_COLOR)

plt.imshow(outputs)
plt.savefig('test.png')


