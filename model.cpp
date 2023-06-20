#include "model.h"
#include "siam_data_set.h"
#include "siam_data_loader.h"

//#define DEBUG

void siam_train(std::vector<std::string> paths_csv, std::string path_save_NN, int epochs, int batch_size, torch::Device device)
{
	if (device == torch::kCPU)
		std::cout << "Training on CPU" << std::endl;
	else
		std::cout << "Training on GPU" << std::endl;

	ConvNet model(3, 64, 64);
	model->to(device);

	Siam_data_set data_set(paths_csv);

	Siam_data_loader data_loader(data_set, batch_size);

	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

	int64_t log_interval = 10;
	int dataset_size = data_set.size();

	float best_mse = std::numeric_limits<float>::max();

	model->train();

	for (int epoch = 1; epoch <= epochs; epoch++) {

		size_t batch_idx = 0;
		float mse = 0.; // mean squared error
		int count = 0;

		for (; !data_loader.epoch_end();) {
			Batch data = data_loader.get_batch();
			
			auto imgs_1 = data.img_1;
			auto imgs_2 = data.img_2;
			auto labels = data.label.squeeze();

			imgs_1 = imgs_1.to(device);
			imgs_2 = imgs_2.to(device);
			labels = labels.to(device);

			optimizer.zero_grad();
			auto output = model->forward(imgs_1, imgs_2);

			//auto loss = torch::binary_cross_entropy_with_logits(output, labels);

			//auto loss = torch::l1_loss(output, labels);

			auto loss = (1 - labels) * torch::pow(output, 2) \
				+ (labels)* torch::pow(torch::clamp(1.0 - output, 0.0), 2);
			//auto loss = labels * torch::pow(output, 2) +
				//(1 - labels) * torch::pow(std::get<1>(torch::max(1.0 - output, 0)), 2);


		
			loss = torch::mean(loss);


#ifdef DEBUG
			std::cout << "siam imgs_1:\n" << imgs_1 << std::endl;
			std::cout << "siam imgs_2:\n" << imgs_2 << std::endl;
			std::cout << "siam labels:\n" << labels << std::endl;
			std::cout << "siam output:\n" << output << std::endl;
			std::cout << "siam loss:\n" << loss << std::endl;
#endif

			loss.backward();
			optimizer.step();

			mse += loss.template item<float>();

			batch_idx++;
			//if (batch_idx % log_interval == 0)
			{
				std::cout << "Train Epoch: " << epoch << "/" << epochs;
				std::cout << " [" << batch_idx * data_loader.size_batch() << "/"  << dataset_size << "]\n" ;
				//std::cout << "Out: " << output;
				//std::cout << "\nLabel: " << labels;
				std::cout << "\nLoss: " << loss.template item<float>() << std::endl;


				//std::printf(
				//	"\rTrain Epoch: %d/%ld [%5ld/%5d] Out:%.4f Loss: %.4f\n",
				//	epoch,
				//	epochs,
				//	batch_idx * data_loader.size_batch(),
				//	dataset_size,
				//	output,
				//	loss.template item<float>());
			}

			count++;
		}

		mse /= (float)count;
		printf(" Mean squared error: %f\n", mse);

		if (mse < best_mse)
		{
			model->to(torch::kCPU);
			model->eval();
			torch::save(model, path_save_NN);
			best_mse = mse;
			std::cout << "model save" << std::endl;
			if (epoch != epochs) {
				model->to(device);
				model->train();
			}
		}
	}
}

void siam_classification(std::string path_img_1, std::string path_img_2, std::string path_NN) {
	auto img_1 = img_to_tensor(path_img_1);
	auto img_2 = img_to_tensor(path_img_2);

	ConvNet model(3, 64, 64);
	torch::load(model, path_NN);

	model->eval();

	torch::Tensor log_prob = model->forward(img_1, img_2);

	if(log_prob.template item<float>() < 0.5)
		std::cout << "identical: " << 1.0 - log_prob.template item<float>() << std::endl;
	else
		std::cout << "no identical: " << log_prob.template item<float>()  << std::endl;
}

void siam_test(std::vector<std::string> paths_csv, std::string path_NN) {
	int error = 0;
	ConvNet model(3, 64, 64);
	torch::load(model, path_NN);

	model->eval();

	Siam_data_set data_set(paths_csv);

	for (int count = 0; count < data_set.size(); count++) {
		auto data = data_set.get(count);
		auto out_model = model->forward(data.img_1, data.img_2);

		if ((out_model.template item<float>() < 0.5 && data.label.template item<int>() != 0)||
			(out_model.template item<float>() >= 0.5 && data.label.template item<int>() != 1)) {
			error++;
		}

		if(count % (data_set.size()/100) == 0)
			std::cout  << count / (data_set.size() / 100)  << " %" << std::endl;
	}

	std::cout << "All: " << data_set.size() << std::endl;
	std::cout << "True: " << data_set.size() - error<< std::endl;
	std::cout << "False: " << error << std::endl;

	std::cout << "Error: " << (double)error/(double)data_set.size() * 100  << std::endl;
}



