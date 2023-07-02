#include "model.h"
#include "siam_data_loader.h"



	int main()
	{
		std::string train_csv = "../siam_data_train.csv"; //path csv file
		std::string val_csv = "../siam_data_val.csv";
		std::string test_csv = "../siam_data_test.csv";

		std::string path_NN = "../best_model.pt"; //path model NN

		auto epochs = 10000;
		auto batch_size = 64;
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

		//siam_train(train_loader, data_set_val, path_NN, epochs);

		ConvNet model(3, 100, 200);
		torch::load(model, path_NN);
		model->eval();

		std::cout <<"Test data ";
		siam_test(data_set_test, model);

		std::cout << "Val data ";
		siam_test(data_set_val, model);
		
		std::cout << "Train data ";
		siam_test(data_set_train, model);

		cv::Mat img = cv::imread("../00000.png");
		std::string work_path = "../work_path";

		int class_img = torch::argmin(siam_classification(img, work_path)).template item<int>();

		std::cout << "Class img: " << class_img << std::endl;

		return 0;
	}