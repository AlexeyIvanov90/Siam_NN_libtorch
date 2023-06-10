#include "model.h"

#define SIAM

	int main()
	{
		//for (int count = 0; count < data_set.size();count++)
			//data_set.get(count).print();

		std::string file_csv = "../file_names.csv";
		std::string path_NN = "../best_model.pt";
		std::string path_img_1 = "../26.png";
		std::string path_img_2 = "../61.png";

		auto epochs = 1;
		auto device = torch::kCPU;

		if (torch::cuda::is_available()) {
			std::cout << "CUDA is available!" << std::endl;
			device = torch::kCUDA;
		}

		device = torch::kCPU;
#ifdef SIAM
		siam_train(file_csv, path_NN, epochs, device);
		siam_classification(path_img_1, path_img_2, path_NN);
		siam_classification(path_img_1, path_img_1, path_NN);
#else
		train(file_csv, path_NN, epochs, device);
		classification(path_img_1, path_NN);
		classification(path_img_2, path_NN);
#endif

		return 0;
	}
