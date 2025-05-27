#pragma once

#include <vector> // Include the vector header
#include <opencv2/core.hpp> // Include OpenCV core header for cv::Mat

struct SegmentationResult {
    std::vector<cv::Mat> masks; // Segmentation masks for each object
    std::vector<int> labels;    // Labels corresponding to each mask
    std::vector<float> scores;  // Confidence scores for each segmentation
    int width;                  // Width of the input image
    int height;                 // Height of the input image

    SegmentationResult() : width(0), height(0) {}
};