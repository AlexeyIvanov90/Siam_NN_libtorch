#include "siam_data_set.h"
#include "data_set.h"
#include "utility.h"



auto siam_read_csv(const std::string& location) -> std::vector<Siam_element> {
	std::fstream in(location, std::ios::in);
	std::string line;
	std::vector<Siam_element> csv;
	std::string label;

	while (getline(in, line))
	{
		Siam_element buf;
		std::stringstream s(line);
		getline(s, buf.img_1, ',');
		getline(s, buf.img_2, ',');
		getline(s, label, ',');

		buf.label = std::stoi(label);

		csv.push_back(buf);
	}
	return csv;
}


Siam_data_set::Siam_data_set(std::string paths_csv)
{
	_data = siam_read_csv(paths_csv);
}


void Siam_data_set::get_img(size_t index) {
	Siam_element obj = _data.at(index);
	obj.print();
}


Siam_element_data Siam_data_set::get(size_t index) {
	if (data_in_ram)
		return _data_mem.at(index);

	auto obj = _data.at(index);

	torch::Tensor img_1 = img_to_tensor(obj.img_1);
	torch::Tensor img_2 = img_to_tensor(obj.img_2);
	torch::Tensor label = torch::full({ 1 }, obj.label);
	label.to(torch::kInt64);

	return Siam_element_data(img_1, img_2, label);
}


size_t Siam_data_set::size() {
	return _data.size();
}


void Siam_data_set::load_to_mem() {
	for(int i = 0; i < _data.size(); i++)
		_data_mem.push_back(get(i));
	data_in_ram = true;
}