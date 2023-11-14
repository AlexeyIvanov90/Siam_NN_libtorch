#ifndef UTIL_H
#define UTIL_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

torch::Tensor img_to_tensor(cv::Mat scr);
torch::Tensor img_to_tensor(std::string file_location);

#endif