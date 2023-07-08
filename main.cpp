#include "model.h"
#include "siam_data_loader.h"
#include "data_set.h"

int main()
{
	std::string siam_train_csv = "../siam_data_train.csv"; 
	std::string siam_val_csv = "../siam_data_val.csv";
	std::string siam_test_csv = "../siam_data_test.csv";

	std::string train_csv = "../data_train.csv";
	std::string val_csv = "../data_val.csv";
	std::string test_csv = "../data_test.csv";

	std::string path_NN = "../best_model.pt"; //path model NN

	cv::Mat img = cv::imread("../00000.png");
	std::string work_path = "../work_path";

	auto epochs = 100000;
	auto batch_size = 1;
	auto device = torch::kCPU;

	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available!" << std::endl;
		device = torch::kCUDA;
	}

	device = torch::kCPU;


	Siam_data_set data_set_siam_train(siam_train_csv);
	data_set_siam_train.load_to_mem();

	Siam_data_set data_set_siam_val(siam_val_csv);
	Siam_data_set data_set_siam_test(siam_test_csv);

	Siam_data_loader train_loader(data_set_siam_train, batch_size);
	Siam_data_loader val_loader(data_set_siam_val, batch_size);


	ConvNet model(3, 100, 200);

	siam_train(train_loader, val_loader, model, epochs, device);

	torch::load(model, path_NN);

	std::cout << "Test data error: " << siam_test(data_set_siam_test, model) << std::endl;
	std::cout << "Val data " << siam_test(data_set_siam_val, model) << std::endl;
	std::cout << "Train data "<< siam_test(data_set_siam_train, model) << std::endl;

	int class_img = torch::argmin(multy_shot_classificator(img_to_tensor(img), work_path)).template item<int>();
	std::cout << "Class img: " << class_img << std::endl;

	Data_set data_set_test(test_csv);
	std::cout << "Test data multy shot error: " << multy_shot_accuracy(data_set_test, work_path) << std::endl;

	return 0;
}