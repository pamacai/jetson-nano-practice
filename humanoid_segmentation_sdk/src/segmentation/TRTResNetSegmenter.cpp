#include "segmentation/TRTResNetSegmenter.hpp"
#include "segmentation/TensorRTUtils.hpp" // Include the helper functions
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

// Define a global logger
class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity != Severity::kINFO) {
                std::cerr << "[TensorRT] " << msg << std::endl;
            }
        }
    } gLogger;


TRTResNetSegmenter::TRTResNetSegmenter(const std::string& modelPath) : engine(nullptr), runtime(nullptr) {
    initialize(modelPath);
}

TRTResNetSegmenter::~TRTResNetSegmenter() {
    cleanup();
}

bool TRTResNetSegmenter::initialize(const std::string& modelPath) {
    // Load the model and initialize the TensorRT engine
    std::ifstream modelFile(modelPath, std::ios::binary);
    if (!modelFile.good()) {
        std::cerr << "Failed to open model file: " << modelPath << std::endl;
        return false;
    }
    modelFile.seekg(0, std::ios::end);
    size_t modelSize = modelFile.tellg();
    modelFile.seekg(0, std::ios::beg);
    std::vector<char> modelData(modelSize);
    modelFile.read(modelData.data(), modelSize);
    modelFile.close();

    runtime = nvinfer1::createInferRuntime(gLogger); // Replace gLogger with your logger
    engine = createEngine(modelData.data(), modelSize, runtime);
    return engine != nullptr;
}

void TRTResNetSegmenter::cleanup() {
    // Clean up resources
    if (engine) {
        destroyEngine(engine);
        engine = nullptr;
    }
    if (runtime) {
        delete runtime; // Use delete instead of destroy
        runtime = nullptr;
    }
}

SegmentationResult TRTResNetSegmenter::segment(const cv::Mat& inputImage) {
    // Preprocess the input image
    cv::Mat preprocessedImage;
    cv::resize(inputImage, preprocessedImage, cv::Size(224, 224)); // Example size
    preprocessedImage.convertTo(preprocessedImage, CV_32F, 1.0 / 255);

    // Perform inference
    std::vector<float> outputData = doInference(engine, preprocessedImage);

    // Postprocess the output data
    return postprocessOutput(outputData);
}

SegmentationResult TRTResNetSegmenter::postprocessOutput(const std::vector<float>& outputData) {
    SegmentationResult result;
    // Placeholder for postprocessing logic
    return result;
}