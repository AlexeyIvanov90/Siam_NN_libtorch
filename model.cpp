#include "model.h"
#include <chrono>
#include <filesystem>


void siam_train(Siam_data_loader &data_train, Siam_data_set &data_val, ConvNet model, int epochs, torch::Device device)
{
	if (device == torch::kCPU)
		std::cout << "Training on CPU" << std::endl;
	else
		std::cout << "Training on GPU" << std::endl;

	model->to(device);

	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

	int64_t log_interval = 10;
	int dataset_size = data_train.size();

	float best_mse = std::numeric_limits<float>::max();

	model->train();

	for (int epoch = 1; epoch <= epochs; epoch++) {
		auto begin = std::chrono::steady_clock::now();

		float train_mse = 0.;
		int train_count = 0;

		float val_error = 0.;		

		size_t bath_counter = 0;

		for (; !data_train.epoch_end();) {
			std::string consol_text = "\r" + std::to_string((int)(((data_train.num_batch()) / ((float)dataset_size/ data_train.size_batch()))*100)) + "%";
			std::cout << consol_text;

			Batch data = data_train.get_batch();
			
			auto imgs_1 = data.img_1;
			auto imgs_2 = data.img_2;
			auto labels = data.label.squeeze();

			imgs_1 = imgs_1.to(device);
			imgs_2 = imgs_2.to(device);
			labels = labels.to(device);

			auto output = model->forward(imgs_1, imgs_2);

			auto loss = torch::diagonal((1 - labels) * torch::pow(output, 2) + (labels)* torch::pow(torch::clamp(1.0 - output, 0.0), 2));
			loss = torch::mean(loss);

			loss.backward();
			optimizer.step();

			train_mse += loss.template item<float>();
			optimizer.zero_grad();
			train_count++;
		}

		model->eval();
		val_error = siam_test(data_val, model);
		model->train();

		std::cout << "\r";

		train_mse /= (float)train_count;

		auto end = std::chrono::steady_clock::now();
		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
		std::cout << "Time for epoch: " << elapsed_ms.count() << " ms\n";

		std::string stat = "Train Epoch: " + std::to_string(epoch) + "/" + std::to_string(epochs) +	
			" Mean squared error: "  + std::to_string(train_mse) +
			" Validation data " + std::to_string(val_error) + "\n";

		std::string model_file_name = "../models/epoch_" + std::to_string(epoch);

		model->to(torch::kCPU);
		model->eval();

		if (val_error < best_mse)
		{
			stat = stat + "save model\n";
			model_file_name = model_file_name + "_best";

			torch::save(model, "../best_model.pt");
			best_mse = val_error;
		}

		std::ofstream out;
		out.open("../models/stat.txt", std::ios::app);
		if (out.is_open())
			out << stat;
		out.close();

		std::cout << stat;
		torch::save(model, model_file_name +".pt");


		if (epoch != epochs) {
			model->to(device);
			model->train();
		}
	}
}


double siam_test(Siam_data_set data_test, ConvNet model){
	int error = 0;

	for (int i = 0; i < data_test.size(); i++) {
		auto data = data_test.get(i);
		auto out_model = model->forward(data.img_1, data.img_2);

		if ((out_model.template item<double>() < 0.5 && data.label.template item<int>() != 0)||
			(out_model.template item<double>() >= 0.5 && data.label.template item<int>() != 1)) {
			error++;
		}
	}

	return (double)error / data_test.size();
}


torch::Tensor multy_shot_classificator(torch::Tensor src, std::string dir) {
	torch::Tensor out;

	std::filesystem::directory_iterator work_dir(dir);
	size_t count_category = 0;

	ConvNet model(3, 100, 200);
	torch::load(model, "../best_model.pt");
	model->eval();

	for (auto const& dir_entry : work_dir)
	{
		if (dir_entry.is_directory()) {
			torch::Tensor buf;
			int count_img = 0;

			for (auto const& dir_category : std::filesystem::directory_iterator(dir_entry))
			{
				auto img = img_to_tensor(dir_category.path().string());
				auto out_model = model->forward(src, img);

				if (count_img == 0)
					buf = out_model;
				else
					buf = torch::cat({ buf, out_model }, 1);
				count_img++;
			}

			auto mean_error = torch::median(buf).view({1,1});

			if (count_category == 0)
				out = mean_error;
			else
				out = torch::cat({out, mean_error }, 1);

			count_category++;
		}
	}

	return out.softmax(1);
}


double multy_shot_accuracy(Data_set scr, std::string dir){
	size_t error = 0;
	for (int i = 0; i < scr.size(); i++) {
		Element_data obj = scr.get(i);
		int class_img = torch::argmin(multy_shot_classificator(obj.img, dir)).template item<int>();
		//std::cout << "Label: " << obj.label.template item<int>() << " Result NN: " << class_img <<std::endl;
		if (class_img != obj.label.template item<int>())
			error++;
	}

	return (double)error/scr.size();
}
