#pragma once

#include <vector>
#include <tuple>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "utils.h"

class CustomDataset : public torch::data::Dataset<CustomDataset>
{
    private:
        std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> csv_;
		int64_t repeat_label = 0;

		torch::Tensor img_to_tensor(std::string file_location) {
			cv::Mat img = cv::imread(file_location);

			torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte).clone();
			img_tensor = img_tensor.toType(torch::kFloat);
			img_tensor = img_tensor.div(255);
			img_tensor = img_tensor.permute({ 2,0,1 });
			return img_tensor;
		}

    public:
        explicit CustomDataset(std::string& file_names_csv)
            // Load csv file with file locations and labels.
            : csv_(ReadCsv(file_names_csv)) {
        };

        // Override the get method to load custom data.
        torch::data::Example<> get(size_t index) override {
            std::string file_location = std::get<0>(csv_[index]);
            int64_t label = std::get<1>(csv_[index]);

			torch::Tensor img_tensor = img_to_tensor(file_location);

            torch::Tensor label_tensor = torch::full({1}, label);

			label_tensor.to(torch::kInt64);

			return {img_tensor, label_tensor};
        };

        // Override the size method to infer the size of the data set.
        torch::optional<size_t> size() const override {
            return csv_.size();
        };
};
