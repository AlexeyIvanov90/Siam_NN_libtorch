#include "model.h"
#include "siam_data_loader.h"



	int main()
	{
		std::string train_csv = "../siam_data_train.csv"; //path csv file
		std::string val_csv = "../siam_data_val.csv";
		std::string test_csv = "../siam_data_test.csv";

		std::string path_NN = "../best_model.pt"; //path model NN

		auto epochs = 10;
		auto batch_size = 1;
		auto device = torch::kCPU;

		if (torch::cuda::is_available()) {
			std::cout << "CUDA is available!" << std::endl;
			device = torch::kCUDA;
		}

		Siam_data_set data_set_train(train_csv);
		data_set_train.load_to_mem();

		Siam_data_set data_set_val(val_csv);
		Siam_data_set data_set_test(test_csv);

		Siam_data_loader train_loader(data_set_train, batch_size);

		siam_train(train_loader, data_set_val, path_NN, epochs);

		ConvNet model(3, 64, 64);
		torch::load(model, path_NN);
		model->eval();

		std::cout <<"Test data ";
		siam_test(data_set_test, model);
		
		return 0;
	}
