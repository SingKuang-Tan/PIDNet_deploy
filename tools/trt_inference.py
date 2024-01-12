import tools._init_paths 
import tensorrt as trt
import torch
from collections import OrderedDict, namedtuple
import numpy as np
import time 
import cv2

import matplotlib.pyplot as plt

from configs import constants as C

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

def input_transform(image):
	mean = [0.485, 0.456, 0.406]
	std=[0.229, 0.224, 0.225]
	image = image.astype(np.float32)[:, :, ::-1] #float32
	image = image / 255.0
	image -= mean
	image /= std
	return image

def trt_inference(trt_file_path):  
    input_image_file_path='/home/monarchtractor/linlin_pidnet/test_images/16.png'
    ori_image = cv2.imread(input_image_file_path)

    transformed_image = input_transform(ori_image)
    #transformed_image=ori_image/255
    #transformed_image=ori_image
    transformed_image = transformed_image.transpose((2, 0, 1)).copy()
    input = torch.from_numpy(transformed_image).unsqueeze(0)#.cuda()
    input=input.repeat(10,1,1,1).cuda()
    print(np.shape(input))
    #exit(0)

    device = 'cuda'
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, namespace="")

    with open(trt_file_path, "rb") as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())

    Binding = namedtuple(
        "Binding", ("name", "dtype", "shape", "data", "ptr"))

    bindings = OrderedDict()
    print('bindings:', model.num_bindings)
    # TensorRT 8.0.1, Jetson Xavier 
    for index in range(model.num_bindings):
        name = model.get_binding_name(index)
        print(name)
        # dtype = trt.nptype(model.get_tensor_dtype(name))
        shape = tuple(model.get_binding_shape(name))
        print(shape)

        data = torch.from_numpy(
            np.empty(shape, dtype=np.float32) #float32
        ).to(device)

        bindings[name] = Binding(
            name, 
            'np.float32', #float32
            shape, 
            data, 
            int(data.data_ptr())
        )

    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())

    context = model.create_execution_context()

    bs = 1
    test_image_t = input #torch.ones(bs, 3, 1024, 1024).to("cuda") #input.to('cuda')
    #test_voxel_t = torch.ones(bs, 1, 200, 200, 50).to("cuda")
    binding_addrs['input'] = int(test_image_t.data_ptr()) #images
    #binding_addrs['voxels'] = int(test_voxel_t.data_ptr())
    context.execute_v2(list(binding_addrs.values()))
    result = bindings['output'].data
    print(result)

    print(np.shape(result))
    #result=result[0]
    result=result.cpu()
    #print(np.shape(result))
    result = np.argmax(result, axis=1).squeeze()
    result = np.array(result.squeeze())
    result = convert_color(result, C.LABEL_TO_COLOR)
    plt.imshow(result)
    plt.savefig('test.png')
    #exit(0)

    print("Testing inference speed.... ")
    print("Warmup 2 inferences with no timing")
    for _ in range(2):
        test_image_t = input #torch.ones(bs, 3, 1024, 1024).to("cuda")
        #test_voxel_t = torch.ones(bs, 1, 200, 200, 50).to("cuda")
        binding_addrs['images'] = int(test_image_t.data_ptr())
        #binding_addrs['voxels'] = int(test_voxel_t.data_ptr())
        context.execute_v2(list(binding_addrs.values()))
        result = bindings['output'].data

    print("Start timing.... ")
    print("Timing 10 inferences")
    start_time = time.time()
    for _ in range(10):
        test_image_t = input #torch.ones(bs, 3, 1024, 1024).to("cuda")
        #test_voxel_t = torch.ones(bs, 1, 200, 200, 50).to("cuda")
        binding_addrs['images'] = int(test_image_t.data_ptr())
        #binding_addrs['voxels'] = int(test_voxel_t.data_ptr())
        context.execute_v2(list(binding_addrs.values()))
        result = bindings['output'].data
    end_time = time.time() - start_time

    print("Time taken for 10 inferences: ", end_time)
    print("Average time per inference: ", end_time/10)


if __name__ == "__main__":
    trt_file_path = './best.trt'
    trt_inference(trt_file_path)