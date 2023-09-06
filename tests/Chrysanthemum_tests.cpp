//
// Created by Joshua Riefman on 2023-09-01.
//

#include "../include/Chrysanthemum.h"
#include "../external/googletest/googletest/include/gtest/gtest.h"

TEST(ChrysanthemumTests, GetNewWeightsTests) {
    std::vector<int> layerSizes_simple = { 2, 2, 2};
    int numInputs_simple = 2;
    const Eigen::Matrix<double, 2, 2> two_x_two { {1, 1}, {1, 1} };
    weights_tensor_t expected_weights_simple = { two_x_two, two_x_two, two_x_two };
    EXPECT_EQ(Chrysanthemum::getNewWeights(layerSizes_simple, numInputs_simple, Chrysanthemum::STANDARD), expected_weights_simple);

    std::vector<int> layerSizes_complex = { 1, 7, 3};
    int numInputs_complex = 10;
    const Eigen::Matrix<double, 1, 10> one_x_ten { {1, 1, 1, 1, 1, 1, 1, 1, 1, 1} };
    const Eigen::Matrix<double, 7, 1> seven_x_one { {1}, {1}, {1}, {1}, {1}, {1}, {1} };
    const Eigen::Matrix<double, 3, 7> three_x_seven { {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1} };
    weights_tensor_t expected_weights_complex = { one_x_ten, seven_x_one, three_x_seven };
    EXPECT_EQ(Chrysanthemum::getNewWeights(layerSizes_complex, numInputs_complex, Chrysanthemum::STANDARD), expected_weights_complex);

    std::vector<int> array_empty {};
    EXPECT_THROW({
        try {
            Chrysanthemum::getNewWeights(array_empty, 1, Chrysanthemum::STANDARD);
        } catch (const std::invalid_argument& exception) {
            EXPECT_STREQ("Invalid arguments provided to getNewWeights!", exception.what());
            throw;
        }
    }, std::invalid_argument) << "Empty array exception not thrown when empty array encountered!";

    EXPECT_THROW({
        try {
            Chrysanthemum::getNewWeights(layerSizes_simple, 0, Chrysanthemum::STANDARD);
        } catch (const std::invalid_argument& exception) {
            EXPECT_STREQ("Invalid arguments provided to getNewWeights!", exception.what());
            throw;
        }
    }, std::invalid_argument) << "Exception not thrown when zero inputs provided!";
}

TEST(ChrysanthemumTests, GetNewBiasesTests) {
    std::vector<int> layerSizes_simple = { 2, 2, 2};
    const Eigen::Vector2d two_vector {1, 1};
    biases_matrix_t expected_biases_simple = { two_vector, two_vector, two_vector };
    EXPECT_EQ(Chrysanthemum::getNewBiases(layerSizes_simple, Chrysanthemum::STANDARD), expected_biases_simple);

    std::vector<int> layerSizes_complex = { 1, 7, 3};
    const Eigen::Matrix<double, 1, 1> one_vector {1};
    const Eigen::Matrix<double, 7, 1> seven_vector { 1, 1, 1, 1, 1, 1, 1 };
    const Eigen::Matrix<double, 3, 1> three_vector { 1, 1, 1 };
    biases_matrix_t expected_biases_complex = { one_vector, seven_vector, three_vector };
    EXPECT_EQ(Chrysanthemum::getNewBiases(layerSizes_complex, Chrysanthemum::STANDARD), expected_biases_complex);

    std::vector<int> array_empty {};
    EXPECT_THROW({
        try {
            Chrysanthemum::getNewBiases(array_empty, Chrysanthemum::STANDARD);
        } catch (const std::invalid_argument& exception) {
            EXPECT_STREQ("Empty layerSizes provided to getNewBiases!", exception.what());
            throw;
        }
    }, std::invalid_argument) << "Empty array exception not thrown when empty array encountered!";
}