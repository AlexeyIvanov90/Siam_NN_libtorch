#ifndef SIAM_DATA_SET_H
#define SIAM_DATA_SET_H

#include <vector>
#include <tuple>
#include <string>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>


struct Siam_element
{
	Siam_element() {};
	Siam_element(std::string img_1, std::string img_2, int label) :img_1{ img_1 }, img_2{ img_2 }, label{ label } {};

	void print() {
		std::cout << "Siam_element:" << std::endl;
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


struct Siam_element_data
{
	Siam_element_data(torch::Tensor img_1, torch::Tensor img_2, torch::Tensor label) :img_1{ img_1 }, img_2{ img_2 }, label{ label } {};

	void print() {
		std::cout << "Siam_element:" << std::endl;
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
	std::vector<Siam_element> _data;
	std::vector<Siam_element_data> _data_mem;
	bool data_in_ram = false;

public:	
	Siam_data_set(std::string paths_csv);
	Siam_element_data get(size_t index);

	void get_img(size_t index);
	size_t size();

	void load_to_mem();
};

#endif