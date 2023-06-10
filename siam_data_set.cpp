#include "siam_data_set.h"


auto Reads_Csv(std::string& location) -> std::vector<std::string> {

	std::fstream in(location, std::ios::in);
	std::string line;
	std::string name;
	std::vector<std::string> csv;

	while (getline(in, line))
	{
		std::stringstream s(line);
		getline(s, name, ',');
		csv.push_back("../" + name);
	}

	return csv;
}

torch::Tensor img_to_tensor(std::string file_location) {
	cv::Mat img = cv::imread(file_location);

	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte).clone();
	img_tensor = img_tensor.toType(torch::kFloat);
	img_tensor = img_tensor.div(255);
	img_tensor = img_tensor.permute({ 2,0,1 });
	img_tensor = img_tensor.unsqueeze(0);
	return img_tensor;
}

Siam_data_set::Siam_data_set(std::vector<std::string> paths_csv)
{
	for each (auto path in paths_csv)
		path_to_class_img.push_back(Reads_Csv(path));
		ñreate_data();
}

void Siam_data_set::ñreate_data() {
	if (path_to_class_img.size() < 2)
		return;

	data.clear();

	size_t min_size_class = path_to_class_img.at(0).size();
	for each (std::vector<std::string> var in path_to_class_img)
		min_size_class = std::min(min_size_class, var.size());

	for (int count = 0, back_count = min_size_class - 1; count < back_count; count++, back_count--) {

		/*
		cv::Mat img_1 = cv::imread(path_to_class_img.at(0).at(count));
		cv::resize(img_1, img_1, cv::Size(300, 300));
		cv::imshow("1", img_1);
		cv::Mat img_2 = cv::imread(path_to_class_img.at(0).at(back_count));
		cv::resize(img_2, img_2, cv::Size(300, 300));
		cv::imshow("2", img_2);
		cv::waitKey();
		*/

		data.push_back(Element(path_to_class_img.at(0).at(count), path_to_class_img.at(0).at(back_count), 0));
		data.push_back(Element(path_to_class_img.at(1).at(back_count), path_to_class_img.at(0).at(back_count), 1));
		data.push_back(Element(path_to_class_img.at(0).at(count), path_to_class_img.at(1).at(count), 1));
		data.push_back(Element(path_to_class_img.at(1).at(count), path_to_class_img.at(1).at(back_count), 0));

	}

	//std::cout << data.size();
	/*
	for each (auto elem in data)
		elem.print();*/
}

Element_data Siam_data_set::get(size_t index) {
	Element obj = data.at(index);
	torch::Tensor img_1 = img_to_tensor(obj.img_1);
	torch::Tensor img_2 = img_to_tensor(obj.img_2);
	torch::Tensor label = torch::full({ 1 }, obj.label);
	label.to(torch::kInt64);

	Element_data out(img_1, img_2, label);
	return out;
}

size_t Siam_data_set::size() {
	size_t out = data.size();
	return out;
}

Siam_data_set::~Siam_data_set()
{
}