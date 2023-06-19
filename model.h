#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

//#define DEBUG

struct ConvNetImpl : public torch::nn::Module 
{
    ConvNetImpl(int64_t channels, int64_t height, int64_t width) 
		: conv1(torch::nn::Conv2dOptions(3 /*input channels*/, 8 /*output channels*/, 5 /*kernel size*/).stride(2)),
          conv2(torch::nn::Conv2dOptions(8, 16, 3).stride(2)),
          
          n(GetConvOutput(channels, height, width)),
          lin1(n, 256),
          lin2(256, 256 /*number of output classes*/),
		  lin3(256, 1) {

        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("lin1", lin1);
        register_module("lin2", lin2);
		register_module("lin3", lin3);
    };

	torch::Tensor first_forward(torch::Tensor x) {
#ifdef DEBUG
		std::cout << x << std::endl;
#endif

		x = torch::relu(torch::max_pool2d(conv1(x), 2));

#ifdef DEBUG
		std::cout << x << std::endl;
#endif

		x = torch::relu(torch::max_pool2d(conv2(x), 2));

#ifdef DEBUG
		std::cout << x << std::endl;
#endif

		x = x.view({ -1, n });

#ifdef DEBUG
		std::cout << x << std::endl;
#endif

		x = torch::relu(lin1(x));

#ifdef DEBUG
		std::cout << x << std::endl;
#endif
		x = lin2(x);
		//x = torch::sigmoid(x);

		//x = torch::relu(lin2(x));

		return x;
	};

    torch::Tensor forward(torch::Tensor x, torch::Tensor y) {

		x = first_forward(x);
		y = first_forward(y);

#ifdef DEBUG
		std::cout << x << std::endl;
		std::cout << y << std::endl;
#endif
		auto siam = torch::abs(x - y);

		siam = lin3(siam);

		//siam = torch::sigmoid(lin3(siam));

#ifdef DEBUG
		std::cout << "siam:\n" << siam;
#endif
		//siam = torch::sigmoid(lin3(siam));
		//siam = torch::relu(lin2(siam));

#ifdef DEBUG
		std::cout << "siam:\n" << siam;
#endif
		return siam;
    };

    // Get number of elements of output.
    int64_t GetConvOutput(int64_t channels, int64_t height, int64_t width) {

        torch::Tensor x = torch::zeros({1, channels, height, width});
        x = torch::max_pool2d(conv1(x), 2);
        x = torch::max_pool2d(conv2(x), 2);

        return x.numel();
    }


    torch::nn::Conv2d conv1, conv2;
    int64_t n;
    torch::nn::Linear lin1, lin2, lin3;
};

TORCH_MODULE(ConvNet);

void classification(std::string path, std::string path_NN);
void train(std::string file_names_csv, std::string path_NN, int epochs, torch::Device device = torch::kCPU);

void siam_classification(std::string path_img_1, std::string path_img_2, std::string path_NN);
void siam_train(std::vector<std::string> paths_csv, std::string path_save_NN, int epochs, int batch_size, torch::Device device = torch::kCPU);
void siam_test(std::vector<std::string> paths_csv, std::string path_NN);