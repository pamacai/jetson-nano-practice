#ifndef TRTRESNETSEGMENTER_HPP
#define TRTRESNETSEGMENTER_HPP

#include "Segmenter.hpp"
#include "SegmentationResult.hpp"
#include <NvInfer.h>
#include <string>

class TRTResNetSegmenter : public Segmenter {
public:
    explicit TRTResNetSegmenter(const std::string& modelPath); // Declaration only
    virtual ~TRTResNetSegmenter();

    bool initialize(const std::string& modelPath) override;
    void cleanup() override;
    SegmentationResult segment(const cv::Mat& inputImage) override;

private:
    nvinfer1::ICudaEngine* engine; // TensorRT engine
    nvinfer1::IRuntime* runtime;   // TensorRT runtime
    void preprocessInput(const cv::Mat& inputImage, float* inputData);
    SegmentationResult postprocessOutput(const std::vector<float>& outputData);
};

#endif


