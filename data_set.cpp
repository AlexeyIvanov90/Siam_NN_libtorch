#include "data_set.h"
#include "siam_data_set.h"


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

torch::Tensor img_to_tensor(cv::Mat scr) {
	cv::cvtColor(scr, scr, CV_BGR2RGB);
	torch::Tensor img_tensor = torch::from_blob(scr.data, { scr.rows, scr.cols, 3 }, torch::kByte).clone();
	img_tensor = img_tensor.toType(torch::kFloat);
	img_tensor = img_tensor.div(255);
	img_tensor = img_tensor.permute({ 2,0,1 });
	img_tensor = img_tensor.unsqueeze(0);
	return img_tensor;
}


torch::Tensor img_to_tensor(std::string file_location) {
	cv::Mat img = cv::imread(file_location);
	return img_to_tensor(img);
}