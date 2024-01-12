import tensorrt as trt
import numpy as np
import os

TRT_LOGGER = trt.Logger()

def build_engine(
    onnx_model_path, 
    tensorrt_engine_path, 
):
    
    # Builder
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    # for Xavier max_workspace_size = 1536
    config.max_workspace_size = 1536 << 20
    # Set FP16 
    config.set_flag(trt.BuilderFlag.FP16)#FP16

    # Enable the INT8 flag in the builder configuration
    # config.set_flag(trt.BuilderFlag.INT8)

    # Create an Int8EntropyCalibrator and set it in the builder configuration
    # calibrator = YourInt8EntropyCalibrator(your_data_samples)
    # config.int8_calibrator = calibrator

    # Onnx parser
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnx_model_path):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")
    with open(onnx_model_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))

    # inputTensor = network.get_input(0) 
    # Dynamic batch (min, opt, max)
    # print('inputTensor.name:', inputTensor.name)
    # if dynamic_axes:
    #     profile.set_shape(inputTensor.name, (min_engine_batch_size, img_size[0], img_size[1], img_size[2]), \
    #         (opt_engine_batch_size, img_size[0], img_size[1], img_size[2]), \
    #         (max_engine_batch_size, img_size[0], img_size[1], img_size[2]))
    #     print('Set dynamic')
    # # else:
    # batch_size = 1
    # img_size = (4, 512, 512)
    # vox_size = (1, 400, 200, 50)
    # profile.set_shape(
    #     'images', 
    #     (batch_size, img_size[0], img_size[1], img_size[2])
    # )
    # profile.set_shape(
    #     'voxels',
    #     (batch_size, vox_size[0], vox_size[1], vox_size[2], vox_size[3]) 
    # )
    # config.add_optimization_profile(profile)
    #network.unmark_output(network.get_output(0))

    # Write engine
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(tensorrt_engine_path, "wb") as f:
        f.write(engineString)
            

class Int8EntropyCalibrator2(trt.IInt8EntropyCalibrator2):
    """
    To Research how to do Int8 quantization and calibration
    """
    def __init__(self, dataset, batch_size, cache_file):
        # This batch size must be set to the batch size of the network for inference
        self.batch_size = batch_size
        self.current_index = 0
        self.dataset = dataset
        self.cache_file = cache_file

        # Allocate enough space for a batch of input data
        self.data = np.zeros((batch_size, 3, 224, 224), dtype=np.float32)
        super(Int8EntropyCalibrator2, self).__init__()

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        if self.current_index + self.batch_size > len(self.dataset):
            return None

        # Copy the next batch of data into the allocated space
        self.data = self.dataset[self.current_index:self.current_index+self.batch_size]
        self.current_index += self.batch_size

        return [int(self.data.ctypes.data)]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


if __name__ == "__main__":
    #polygraphy surgeon sanitize best.onnx --fold-constants --output best2.onnx
    
    root_folder = '.'
    onnx_model_name = "best.onnx"
    trt_engine_name = "best.trt"
    
    onnx_model_path = os.path.join(root_folder, onnx_model_name)
    trt_engine_path = os.path.join(root_folder, trt_engine_name)
    build_engine(
        onnx_model_path=onnx_model_path, 
        tensorrt_engine_path=trt_engine_path
    )