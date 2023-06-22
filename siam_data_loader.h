#pragma once

#include "siam_data_set.h"
#include <random> 

struct Batch
{
	Batch(torch::Tensor img_1, torch::Tensor img_2, torch::Tensor  label) :img_1(img_1), img_2(img_2), label(label) {};
	torch::Tensor img_1;
	torch::Tensor img_2;
	torch::Tensor label;
};

class Siam_data_loader
{
private:
	torch::Tensor batch_img_1;
	torch::Tensor batch_img_2;
	torch::Tensor batch_label;

	size_t data_size;
	size_t batch_size;

	size_t count_batch = 0;
	Siam_data_set data;

	std::vector<size_t> random_index;
public:
	Siam_data_loader(Siam_data_set data, size_t batch_size);

	Batch Siam_data_loader::get_batch();
	size_t Siam_data_loader::num_batch();
	size_t size();
	size_t size_batch();
	bool epoch_end();

	void random_data();
};
