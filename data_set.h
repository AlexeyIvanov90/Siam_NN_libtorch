#pragma once

#include <vector>
#include <tuple>
#include <string>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>


struct Element
{
	Element() {};
	Element(std::string img, int label) :img{ img }, label{ label } {};

	void print() {
		std::cout << "Element:" << std::endl;
		cv::Mat img_show = cv::imread(img);

		std::cout << img << std::endl;
		cv::imshow("Img 1", img_show);

		std::cout << label << std::endl;

		cv::waitKey();
	}

	std::string img;
	int label;
};


struct Element_data
{
	Element_data(torch::Tensor img,  torch::Tensor label) :img{ img }, label{ label } {};

	void print() {
		std::cout << "Element:" << std::endl;
		std::cout << img.sizes() << std::endl;
		std::cout << label << std::endl;
	}

	torch::Tensor img;
	torch::Tensor label;
};


class Data_set
{
private:
	std::vector<Element> _data;
	bool data_in_ram = false;

public:
	Data_set(std::string paths_csv);
	Element_data get(size_t index);

	cv::Mat get_img(size_t index);
	size_t size();
};


torch::Tensor img_to_tensor(cv::Mat scr);
torch::Tensor img_to_tensor(std::string file_location);