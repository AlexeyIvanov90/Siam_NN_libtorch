#include "data_set.h"
#include "siam_data_set.h"
#include "utility.h"


auto read_csv(const std::string& location) -> std::vector<Element> {
	std::fstream in(location, std::ios::in);
	std::string line;
	std::vector<Element> csv;
	std::string label;

	while (getline(in, line))
	{
		Element buf;
		std::stringstream s(line);
		getline(s, buf.img, ',');
		getline(s, label, ',');

		buf.label = std::stoi(label);

		csv.push_back(buf);
	}
	return csv;
}


Data_set::Data_set(std::string paths_csv)
{
	_data = read_csv(paths_csv);
}


cv::Mat Data_set::get_img(size_t index) {
	Element obj = _data.at(index);
	return cv::imread(obj.img);
}


Element_data Data_set::get(size_t index) {
	auto obj = _data.at(index);

	torch::Tensor img = img_to_tensor(obj.img);
	torch::Tensor label = torch::full({ 1 }, obj.label);
	label.to(torch::kInt64);

	return Element_data(img, label);
}


size_t Data_set::size() {
	return _data.size();
}