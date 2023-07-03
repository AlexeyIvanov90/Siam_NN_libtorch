#include "siam_data_loader.h"

Siam_data_loader::Siam_data_loader(Siam_data_set &data, size_t batch_size) :data(data), batch_size(batch_size), data_size(data.size()) {
	random_index.resize(data_size);
	for (size_t index = 0; index < random_index.size(); index++)
		random_index.at(index) = index;

	random_data();
}

void Siam_data_loader::random_data() {
	auto rng = std::default_random_engine{ rd() };
	std::shuffle(random_index.begin(), random_index.end(), rng);
}

Batch Siam_data_loader::get_batch() {
	bool flag = false;

	for (; count_batch < data_size; ) {
		if (count_batch%batch_size == 0) {
			Element_data x = data.get(random_index.at(count_batch));

			batch_img_1 = x.img_1;
			batch_img_2 = x.img_2;
			batch_label = x.label;
		}
		else {
			Element_data x = data.get(random_index.at(count_batch));

			batch_img_1 = torch::cat({ batch_img_1, x.img_1 }, 0);
			batch_img_2 = torch::cat({ batch_img_2, x.img_2 }, 0);
			batch_label = torch::cat({ batch_label, x.label }, 0);

		}
		count_batch++;
		if (count_batch%batch_size == 0) {
			break;
		}
	}

	return Batch(batch_img_1, batch_img_2, batch_label);
}

bool Siam_data_loader::epoch_end() {
	if (count_batch == data_size) {
		count_batch = 0;
		random_data();
		return true;
	}
	else {
		return false;
	}
}

size_t Siam_data_loader::size_batch() {
	return batch_size;
}

size_t Siam_data_loader::num_batch() {
	return count_batch / batch_size;
}

size_t Siam_data_loader::size() {
	return data_size;
}