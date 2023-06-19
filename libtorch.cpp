#include "model.h"

#define SIAM

	int main()
	{
		std::vector<std::string> paths_csv;// путь к файлам с тренеровачными данными
		paths_csv.push_back("../category_1.csv");
		paths_csv.push_back("../category_2.csv");

		std::vector<std::string> paths_csv_test;// путь к файлам с тестовыми данными
		paths_csv_test.push_back("../category_1_test.csv");
		paths_csv_test.push_back("../category_2_test.csv");


		std::string path_NN = "../best_model.pt";
		std::string path_img_1 = "../1.png";// зерно
		std::string path_img_2 = "../2.png";// зерно
		std::string path_img_3 = "../3.png";// не зерно
		std::string path_img_4 = "../4.png";// не зерно
		std::string path_img_5 = "../5.png";// зерно
		std::string path_img_6 = "../6.png";// зерно


		auto epochs = 50;
		auto batch_size = 1;

		auto device = torch::kCPU;

		if (torch::cuda::is_available()) {
			std::cout << "CUDA is available!" << std::endl;
			device = torch::kCUDA;
		}

		device = torch::kCPU;
#ifdef SIAM
		//siam_train(paths_csv, path_NN, epochs, batch_size);

		//siam_test(paths_csv, path_NN);

		
		siam_classification(path_img_1, path_img_2, path_NN);//0
		siam_classification(path_img_3, path_img_4, path_NN);//0

		siam_classification(path_img_1, path_img_3, path_NN);//1
		siam_classification(path_img_1, path_img_4, path_NN);//1

		siam_classification(path_img_2, path_img_3, path_NN);//1
		siam_classification(path_img_2, path_img_4, path_NN);//1

		siam_classification(path_img_5, path_img_6, path_NN);//0


#else
		train(file_csv, path_NN, epochs, device);
		classification(path_img_1, path_NN);
		classification(path_img_2, path_NN);
#endif

		return 0;
	}
