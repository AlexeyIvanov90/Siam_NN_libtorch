#include "utility.h"

torch::Tensor img_to_tensor(cv::Mat scr) {
	cv::cvtColor(scr, scr, CV_BGR2RGB);
	torch::Tensor img_tensor = torch::from_blob(scr.data, { scr.rows, scr.cols, 3 }, torch::kByte).clone();
	img_tensor = img_tensor.toType(torch::kFloat);
	img_tensor = img_tensor.div(255);
	img_tensor = img_tensor.permute({ 2,0,1 });
	img_tensor = img_tensor.unsqueeze(0);
	return img_tensor;
}


torch::Tensor img_to_tensor(std::string file_location) {
	cv::Mat img = cv::imread(file_location);
	return img_to_tensor(img);
}