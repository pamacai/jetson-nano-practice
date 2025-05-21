import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import onnx

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

onnx_model_path = "resnet50.onnx"
try:
    model = onnx.load(onnx_model_path)
    onnx.checker.check_model(model)
    print("ONNX model is valid.")
except Exception as e:
    print(f"ONNX model validation failed: {e}")

import tensorrt as trt
print(trt.__version__)
print(dir(trt.ICudaEngine))

def preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = img.transpose((2, 0, 1))  # Convert HWC to CHW
    return img[np.newaxis, :]  # Add batch dimension

def load_engine(onnx_file_path):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        # Create a builder configuration
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # Set workspace size to 1GB

        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Check for dynamic input shapes and create an optimization profile
        input_tensor = network.get_input(0)
        if input_tensor.shape[0] == -1:  # Dynamic batch size
            profile = builder.create_optimization_profile()
            profile.set_shape(input_tensor.name, (1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224))
            config.add_optimization_profile(profile)

        # Build serialized engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Failed to build the serialized engine.")
            return None

        # Deserialize the engine
        with trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(serialized_engine)

def infer(engine, input_tensor):
    import numpy as np
    import pycuda.driver as cuda
    import time

    h_input = np.ascontiguousarray(input_tensor)
    context = engine.create_execution_context()

    # Find input and output tensor names
    input_name = None
    output_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
        elif engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            output_name = name

    if input_name is None or output_name is None:
        print("Could not find input or output tensor names.")
        return None

    # Set input shape
    context.set_input_shape(input_name, h_input.shape)

    # Allocate device memory
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(1000 * 4)  # Use actual output size if known

    # Copy input to device
    cuda.memcpy_htod(d_input, h_input)

    # Set tensor addresses
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # Run inference
    start = time.time()
    # ... (set up context, set input shape, set tensor addresses, etc.)

    stream = cuda.Stream()
    context.execute_async_v3(stream.handle)

    cuda.Context.synchronize()

    # context.execute_v3()
    # cuda.Context.synchronize()
    end = time.time()

    # Copy output back to host
    h_output = np.empty([1000], dtype=np.float32)
    cuda.memcpy_dtoh(h_output, d_output)

    print(f"Inference time: {(end - start)*1000:.2f} ms")
    return h_output


if __name__ == "__main__":
    onnx_model_path = "resnet50.onnx"
    image_path = "the_dog1.jpeg"

    input_tensor = preprocess(image_path)
    engine = load_engine(onnx_model_path)
    output = infer(engine, input_tensor)

    top5 = np.argsort(output)[-5:][::-1]
    print("Top-5 predictions:", top5)
