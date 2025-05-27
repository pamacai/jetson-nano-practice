#ifndef SEGMENTER_HPP
#define SEGMENTER_HPP

#include "SegmentationResult.hpp"

class Segmenter {
public:
    virtual ~Segmenter() {}

    virtual bool initialize(const std::string& modelPath) = 0;
    virtual SegmentationResult segment(const cv::Mat& inputImage) = 0;
    virtual void cleanup() = 0;
};

#endif // SEGMENTER_HPP