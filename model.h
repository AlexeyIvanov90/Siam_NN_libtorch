#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>

class Data_set;
class Siam_data_set;
class Siam_data_loader;

struct BasicBlock : torch::nn::Module {

	static const int expansion;

	int64_t stride;
	torch::nn::Conv2d conv1;
	torch::nn::BatchNorm2d bn1;
	torch::nn::Conv2d conv2;
	torch::nn::BatchNorm2d bn2;
	torch::nn::Sequential downsample;

	BasicBlock(int64_t inplanes, int64_t planes, int64_t stride_ = 1,
		torch::nn::Sequential downsample_ = torch::nn::Sequential());

	torch::Tensor forward(torch::Tensor x);
};

struct ConvNetImpl : public torch::nn::Module 
{
	ConvNetImpl(int64_t channels, int64_t height, int64_t width);
	torch::Tensor first_forward(torch::Tensor x);
	torch::Tensor forward(torch::Tensor x, torch::Tensor y);

	int64_t GetConvOutput(int64_t channels, int64_t height, int64_t width);

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


void siam_train(Siam_data_loader &data_train, Siam_data_set &data_val, ConvNet model, int epochs, torch::Device device = torch::kCPU);
double siam_test(Siam_data_set data_test, ConvNet model);
torch::Tensor multy_shot_classificator(torch::Tensor src, std::string dir_model);
double multy_shot_accuracy(Data_set scr, std::string dir);

#endif