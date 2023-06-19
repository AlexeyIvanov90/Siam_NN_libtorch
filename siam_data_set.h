#pragma once

#include <vector>
#include <tuple>
#include <string>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

struct Element
{
	Element(std::string img_1, std::string img_2, int label) :img_1{ img_1 }, img_2{ img_2 }, label{ label } {};

	void print() {
		std::cout << "Element:" << std::endl;
		cv::Mat img_1_show = cv::imread(img_1);
		cv::Mat img_2_show = cv::imread(img_2);

		std::cout << img_1 << std::endl;
		cv::imshow("Img 1", img_1_show);

		std::cout << img_2 << std::endl;
		cv::imshow("Img 2", img_2_show);

		std::cout << label << std::endl;

		cv::waitKey();
	}

	std::string img_1;
	std::string img_2;
	int label;
};

struct Element_data
{
	Element_data(torch::Tensor img_1, torch::Tensor img_2, torch::Tensor label) :img_1{ img_1 }, img_2{ img_2 }, label{ label } {};

	void print() {
		std::cout << "Element:" << std::endl;
		std::cout << img_1.sizes() << std::endl;
		std::cout << img_2.sizes() << std::endl;
		std::cout << label << std::endl;
	}

	torch::Tensor img_1;
	torch::Tensor img_2;
	torch::Tensor label;
};

class Siam_data_set
{
private:
	std::vector<Element> data; // готовый датасет
	std::vector < std::vector<std::string>> path_to_class_img; // 
public:
	Siam_data_set(std::vector<std::string> path_csv);
	~Siam_data_set();
	void сreate_data();
	Element_data get(size_t index);
	size_t size();
};

torch::Tensor img_to_tensor(std::string file_location);
