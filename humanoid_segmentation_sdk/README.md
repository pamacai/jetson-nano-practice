# Humanoid Segmentation SDK

The Humanoid Segmentation SDK is a software development kit designed for performing segmentation tasks using a TensorRT-based model. This SDK provides an interface for segmenting images and includes a concrete implementation that utilizes a pre-trained ONNX model.

## Project Structure

The project is organized as follows:

```
humanoid_segmentation_sdk/
├── CMakeLists.txt                # Build configuration file
├── include/
│   └── segmentation/
│       ├── Segmenter.hpp         # Abstract interface for segmenters
│       ├── SegmentationResult.hpp # Output data structure for segmentation results
│       └── TRTResNetSegmenter.hpp # Concrete implementation of the Segmenter interface
├── src/
│   └── segmentation/
│       ├── TRTResNetSegmenter.cpp # Implementation of the TRTResNetSegmenter class
│       └── cuda_kernels.cu       # CUDA kernels for postprocessing
├── models/
│   └── resnet18-seg.onnx         # ONNX model file for segmentation
├── data/
│   └── example.jpg               # Example image for testing
├── tools/
│   └── demo_runner.cpp           # Example usage of the SDK
├── scripts/
│   └── build_jetson.sh           # Build script for Jetson devices
├── doc/
│   └── conan_setup.md            # Instructions for setting up Conan
└── README.md                     # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd humanoid_segmentation_sdk
   ```

2. **Install dependencies:**
   - Follow the [Conan setup instructions](doc/conan_setup.md) to configure Conan for dependency management.
   - Ensure you have TensorRT installed on your system. Refer to the [TensorRT setup guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) for detailed installation steps.

3. **Build the project:**
   ```
   unset CROSS_COMPILE
   unset CC
   conan install . --output-folder=build  --build=missing
   <!-- conan install . --output-folder=build --build=missing -o "*:shared=True" -->
   <!-- with :=shared=True does not work -->
   cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
   cmake ~/workspace/jetson-nano-practice/humanoid_segmentation_sdk/
   make -j$(nproc)
   ```

4. **Run the demo:**
   After building, you can run the demo provided in the `tools/demo_runner.cpp` file to see the segmentation in action.

## Usage

To use the SDK, include the necessary headers from the `include/segmentation` directory in your application. Create an instance of the `TRTResNetSegmenter` class and call its methods to perform segmentation on input images.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

This SDK utilizes the TensorRT framework and ONNX models for efficient inference. Special thanks to the contributors and the community for their support.Humanoid Segmentation SDK

