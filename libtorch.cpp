#include "model.h"

void test(torch::DeviceType device) {
	ConvNet model(3, 64, 64);
	model->to(device);
	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

	auto imgs = torch::zeros({ 3,64,64 }, torch::kFloat);
	auto labels = torch::zeros(1, torch::kInt64);

	imgs = imgs.unsqueeze(0);

	imgs = imgs.to(device);
	labels = labels.to(device);

	optimizer.zero_grad();
	auto output = model(imgs);
	auto loss = torch::nll_loss(output, labels);

	loss.backward();
	optimizer.step();

	std::cout << labels << std::endl;
	std::cout << output << std::endl;
}

int main()
{
	std::string file_csv = "../file_names.csv";
	std::string path_NN = "../best_model.pt";
	std::string path_img = "../26.png";

	auto epochs = 2;
	auto device = torch::kCPU;

	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available!" << std::endl;
		device = torch::kCUDA;
	}

	device = torch::kCPU;
	   	 
	train(file_csv, path_NN, epochs, device);
	classification(path_img, path_NN);
	return 0;
}