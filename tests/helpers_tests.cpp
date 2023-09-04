//
// Created by Joshua Riefman on 2023-09-01.
//

#include "../include/helpers.h"
#include "../external/googletest/googletest/include/gtest/gtest.h"

TEST(HelpersTests, RelU_Tests) {
    EXPECT_EQ(helpers::ReLU(10), 10) << "Argument of 10 did not result in 10!";
    EXPECT_EQ(helpers::ReLU(0), 0) << "Argument of 0 did not result in 0!";
    EXPECT_EQ(helpers::ReLU(-10), 0) << "Argument of -10 did not result in 0!";
}

TEST(HelpersTests, MaxInArray_Tests) {
    std::vector<int> array_with_max_5 = {1, 2, 3, 4, 5};
    EXPECT_EQ(helpers::MaxInArray(array_with_max_5), 5) << "Argument with expected value of 5 did not result in 5!";

    std::vector<int> array_with_max_1 = {1, 1, 1, 1, 1};
    EXPECT_EQ(helpers::MaxInArray(array_with_max_1), 1) << "Argument with expected value of 1 did not result in 1!";

    std::vector<int> array_empty {};
    EXPECT_THROW({
            try {
                helpers::MaxInArray(array_empty);
            } catch (const std::invalid_argument& exception) {
                EXPECT_STREQ("Trying to find maximum of empty array!", exception.what());
                throw;
            }
    }, std::invalid_argument) << "Empty array exception not thrown when empty array encountered!";
}

TEST(HelpersTests, Sum_Tests) {
    std::vector<int> array_with_sum_15 = {1, 2, 3, 4, 5};
    EXPECT_EQ(helpers::Sum(array_with_sum_15), 15) << "Argument of expected sum 15 did not result in sum of 15!";

    std::vector<int> array_with_sum_0 = {-1, 1, -2, 2, 0};
    EXPECT_EQ(helpers::Sum(array_with_sum_0), 0) << "Argument of expected sum 15 did not result in sum of 15!";

    std::vector<int> array_empty {};
    EXPECT_THROW({
        try {
            helpers::Sum(array_empty);
        } catch (const std::invalid_argument& exception) {
            EXPECT_STREQ("Trying to find sum of empty array!", exception.what());
            throw;
        }
    }, std::invalid_argument) << "Empty array exception not thrown when empty array encountered!";
}

TEST(HelpersTests, GetRandomNormalized_Tests) {
    const int num_values_to_test = 100;
    double value_1;
    double value_2 = helpers::GetRandomNormalized();
    std::vector<double> possibleValues {};

    for (int i = 0; i < num_values_to_test; i++) {
        value_1 = helpers::GetRandomNormalized();
        EXPECT_NE(value_1, value_2) << "Adjacent random values are identical!";
        ASSERT_LE(value_1, 1) << "Value is not less than 1!";
        ASSERT_GE(value_1, -1) << "Value is not greater than -1!";

        value_2 = value_1;
    }
}

