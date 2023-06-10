#include "siam_data_loader.h"

Siam_data_loader::Siam_data_loader(Siam_data_set data, size_t batch_size) :data(data), batch_size(batch_size), data_size(data.size()) {}

Batch Siam_data_loader::get_batch() {
	std::cout << "batch " << count_batch/batch_size + 1 << 
		"/"<< data_size/batch_size + 1 << std::endl;
	
	for (; count_batch < data_size; ) {
		if (count_batch%batch_size == 0) {
			batch_img_1 = data.get(count_batch).img_1;
			batch_img_2 = data.get(count_batch).img_2;
			batch_label = data.get(count_batch).label;
		}
		else {
			batch_img_1 = torch::cat({ batch_img_1, data.get(count_batch).img_1 }, 0);
			batch_img_2 = torch::cat({ batch_img_2, data.get(count_batch).img_2 }, 0);
			batch_label = torch::cat({ batch_label, data.get(count_batch).label }, 0);
		}
		count_batch++;
		if (count_batch%batch_size == 0) {
			break;
		}
	}

	Batch out(batch_img_1, batch_img_2, batch_label);

	return out;
}

Siam_data_loader::~Siam_data_loader() {}

bool Siam_data_loader::epoch_end() {
	if (count_batch == data_size) {
		count_batch = 0;
		return true;
	}
	else {
		return false;
	}
}

int Siam_data_loader::size_batch() {
	return batch_size;
}
