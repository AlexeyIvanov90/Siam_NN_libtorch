#include "model.h"
#include  "custom_dataset.h"
#include "siam_data_set.h"
#include "siam_data_loader.h"

//#define DEBUG

void train(std::string file_names_csv, std::string path_save_NN, int epochs, torch::Device device)
{
	if (device == torch::kCPU)
		std::cout << "Training on CPU" << std::endl;
	else
		std::cout << "Training on GPU" << std::endl;

	ConvNet model(3, 64, 64);
	model->to(device);

	torch::data::DataLoaderOptions OptionsData;
	OptionsData.batch_size(100).workers(4);

	auto data_set = CustomDataset(file_names_csv).map(torch::data::transforms::Stack<>());

	auto data_loader_1 = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		data_set,
		OptionsData);

	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

	int64_t log_interval = 10;
	int dataset_size = data_set.size().value();

	float best_mse = std::numeric_limits<float>::max();

	model->train();

	for (int epoch = 1; epoch <= epochs; epoch++) {

		size_t batch_idx = 0;
		float mse = 0.; // mean squared error
		int count = 0;

		for (auto& batch : *data_loader_1) {
			auto imgs = batch.data;
			auto labels = batch.target.squeeze();

			imgs = imgs.to(device);
			labels = labels.to(device);

			optimizer.zero_grad();
			auto output = model(imgs, imgs);

			//auto loss = torch::nll_loss(output, labels);

			auto loss = torch::l1_loss(output, labels);

			/*
			std::cout << "imgs:\n" << imgs.sizes() << std::endl;
			std::cout << "imgs:\n" << imgs.sizes() << std::endl;
			std::cout << "labels:\n" << labels.sizes() << std::endl;
			std::cout << "output:\n" << output << std::endl;
			*/

			loss.backward();
			optimizer.step();

			mse += loss.template item<float>();

			batch_idx++;
			if (batch_idx % log_interval == 0)
			{
				std::printf(
					"\rTrain Epoch: %d/%ld [%5ld/%5d] Loss: %.4f",
					epoch,
					epochs,
					batch_idx * batch.data.size(0),
					dataset_size,
					loss.template item<float>());
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

void siam_train(std::string file_names_csv, std::string path_save_NN, int epochs, torch::Device device)
{
	if (device == torch::kCPU)
		std::cout << "Training on CPU" << std::endl;
	else
		std::cout << "Training on GPU" << std::endl;

	ConvNet model(3, 64, 64);
	model->to(device);

	std::vector<std::string> paths_csv;
	paths_csv.push_back("../category_1.csv");
	paths_csv.push_back("../category_2.csv");

	Siam_data_set data_set(paths_csv);

	Siam_data_loader data_loader(data_set, 10);

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
			auto loss = torch::l1_loss(output, labels);

#ifdef DEBUG
			std::cout << "siam imgs_1:\n" << imgs_1 << std::endl;
			std::cout << "siam imgs_2:\n" << imgs_2 << std::endl;
			std::cout << "siam labels:\n" << labels << std::endl;
			std::cout << "siam output:\n" << output << std::endl;
			std::cout << "siam loss:\n" << loss << std::endl;
#endif
			std::cout << "siam output:\n" << output << std::endl;

			loss.backward();
			optimizer.step();

			mse += loss.template item<float>();

			batch_idx++;
			if (batch_idx % log_interval == 0)
			{
				std::printf(
					"\rTrain Epoch: %d/%ld [%5ld/%5d] Out:%.4f Loss: %.4f\n",
					epoch,
					epochs,
					batch_idx * data_loader.size_batch(),
					dataset_size,
					output,
					loss.template item<float>());
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

void classification(std::string path, std::string path_NN)
{
	std::string loc = path;

	cv::Mat img = cv::imread(loc);
	cv::imshow("", img);
	cv::waitKey();

	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte).clone();
	img_tensor = img_tensor.toType(torch::kFloat);
	img_tensor = img_tensor.div(255);
	img_tensor = img_tensor.permute({ 2,0,1 });
	img_tensor = img_tensor.unsqueeze(0);

	ConvNet model(3, 64, 64);
	torch::load(model, path_NN);

	torch::Tensor log_prob = model(img_tensor, img_tensor);
	//torch::Tensor log_prob = model->forward(img_tensor);
	//torch::Tensor prob = torch::exp(log_prob);

	printf("Probability of being\n\
    an zerno = %.2f \n", log_prob);

	/*
	printf("Probability of being\n\
	an zerno = %.2f percent\n\
	a zerno primes = %.2f percent\n", prob[0][0].item<float>()*100., prob[0][1].item<float>()*100.);*/
}

void siam_classification(std::string path_img_1, std::string path_img_2, std::string path_NN) {
	auto img_1 = img_to_tensor(path_img_1);
	auto img_2 = img_to_tensor(path_img_2);

	ConvNet model(3, 64, 64);
	torch::load(model, path_NN);

	torch::Tensor log_prob = model(img_1, img_2);

	std::cout << log_prob << std::endl;
}



