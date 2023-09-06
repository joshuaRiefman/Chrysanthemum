//
// Created by Joshua Riefman on 2023-09-01.
//
#include "../include/NeuralNetwork.h"
#include "../external/googletest/googletest/include/gtest/gtest.h"

TEST(NeuralNetworkTests, NeuralNetworkConstructionTests) {
    const int numInputs = 2;
    const Eigen::Matrix<double, 2, 2> two_x_two { {1, 1}, {1, 1} };
    const Eigen::Vector2d two_vector {1, 1};
    const std::vector<int> layerSizes = {2, 2, 2};

    weights_tensor_t weights_simple = { two_x_two, two_x_two, two_x_two };
    biases_matrix_t biases_simple = { two_vector, two_vector, two_vector };

    std::unique_ptr<weights_tensor_t> weights = std::make_unique<weights_tensor_t>(weights_simple);
    std::unique_ptr<biases_matrix_t> biases = std::make_unique<biases_matrix_t>(biases_simple);

    EXPECT_NO_THROW(NeuralNetwork(numInputs, layerSizes, weights, biases)) << "Valid configuration triggered exception!";

    const std::vector<int> layerSizes_invalid = {1, 2, 3};
    const int numInputs_invalid = 2;

    EXPECT_THROW({
        try {
            NeuralNetwork(numInputs, layerSizes_invalid, weights, biases);
        } catch (const std::invalid_argument& exception) {
            throw;
        }
    }, std::invalid_argument) << "Invalid layerSizes not caught when provided to NN!";

    EXPECT_THROW({
        try {
            NeuralNetwork(numInputs_invalid, layerSizes, weights, biases);
        } catch (const std::invalid_argument& exception) {
            throw;
        }
    }, std::invalid_argument) << "Invalid numInputs not caught when provided to NN!";
}

TEST(NeuralNetworkTests, SolveTests) {
    const std::vector<double> inputs {1, 1};
    const Eigen::Matrix<double, 2, 2> two_x_two { {1, 1}, {1, 1} };
    const Eigen::Vector2d two_vector {1, 1};
    const std::vector<int> layerSizes = {2, 2, 2};

    weights_tensor_t weights_simple = { two_x_two, two_x_two, two_x_two };
    biases_matrix_t biases_simple = { two_vector, two_vector, two_vector };

    std::unique_ptr<weights_tensor_t> weights = std::make_unique<weights_tensor_t>(weights_simple);
    std::unique_ptr<biases_matrix_t> biases = std::make_unique<biases_matrix_t>(biases_simple);

    NeuralNetwork neuralNetwork((int)inputs.size(), layerSizes, weights, biases);

    std::vector<double> expected_outputs_to_1_1 {15, 15};
    EXPECT_EQ(neuralNetwork.solve({1, 1}), expected_outputs_to_1_1) << "NN did not return expected output!";
    std::vector<double> expected_outputs_to_0_0 {7, 7};
    EXPECT_EQ(neuralNetwork.solve({0, 0}), expected_outputs_to_0_0) << "NN did not return expected output!";
}

