#pragma once

#include "siam_data_set.h"
#include "siam_data_loader.h"

struct ConvNetImpl : public torch::nn::Module 
{
    ConvNetImpl(int64_t channels, int64_t height, int64_t width) 
		: conv1(torch::nn::Conv2dOptions(3 /*input channels*/, 8 /*output channels*/, 5 /*kernel size*/).stride(2)),
		bn2d_1(torch::nn::BatchNorm2d(8)),
        conv2(torch::nn::Conv2dOptions(8, 16, 3).stride(2)),
		bn2d_2(torch::nn::BatchNorm2d(16)),
        n(GetConvOutput(channels, height, width)),
        lin1(n, 64),
        lin2(64, 64),
		lin3(64, 1) {

        register_module("conv1", conv1);
		register_module("bn2d_1", bn2d_1);
        register_module("conv2", conv2);
		register_module("bn2d_2", bn2d_2);
        register_module("lin1", lin1);
        register_module("lin2", lin2);
		register_module("lin3", lin3);
    };

	torch::Tensor first_forward(torch::Tensor x) {
		x = torch::relu(torch::max_pool2d(conv1(x), 2));
		x = torch::batch_norm(bn2d_1->forward(x), bn1W, bnBias1W, bnmean1W, bnvar1W, true, 0.9, 0.001, true);
		x = torch::relu(torch::max_pool2d(conv2(x), 2));
		x = torch::batch_norm(bn2d_2->forward(x), bn2W, bnBias2W, bnmean2W, bnvar2W, true, 0.9, 0.001, true);

		x = x.view({ -1, n });

		x = torch::relu(lin1(x));

		x = torch::relu(lin2(x));

		//x = lin2(x);

		return x;
	};

    torch::Tensor forward(torch::Tensor x, torch::Tensor y) {

		x = first_forward(x);
		y = first_forward(y);

		x = torch::abs(x - y);

		x = lin3(x);

		x = torch::sigmoid(x);

		return x;
    };

    // Get number of elements of output.
    int64_t GetConvOutput(int64_t channels, int64_t height, int64_t width) {

        torch::Tensor x = torch::zeros({1, channels, height, width});
        x = torch::max_pool2d(conv1(x), 2);
        x = torch::max_pool2d(conv2(x), 2);

        return x.numel();
    }


    torch::nn::Conv2d conv1, conv2;
	torch::nn::BatchNorm2d bn2d_1, bn2d_2;
    int64_t n;
    torch::nn::Linear lin1, lin2, lin3;

	torch::Tensor bn1W;
	torch::Tensor bnBias1W;
	torch::Tensor bnmean1W;
	torch::Tensor bnvar1W;

	torch::Tensor bn2W;
	torch::Tensor bnBias2W;
	torch::Tensor bnmean2W;
	torch::Tensor bnvar2W;
};

TORCH_MODULE(ConvNet);

void siam_classification(std::string path_img_1, std::string path_img_2, std::string path_NN);
void siam_train(Siam_data_loader data_train, Siam_data_set data_val, std::string path_save_NN, int epochs, torch::Device device = torch::kCPU);
void siam_test(Siam_data_set data_test, ConvNet model);