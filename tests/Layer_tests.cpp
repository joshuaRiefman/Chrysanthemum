//
// Created by Joshua Riefman on 2023-09-01.
//

#include "../include/Layer.h"
#include "../external/googletest/googletest/include/gtest/gtest.h"

TEST(LayerTests, EvaluationTests) {
    {
        const long numInputs = 3;
        const long numOutputs = 2;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> weights {
                {1, 1, 1},
                {1, 1, 1}
        };
        Eigen::VectorXd biases {{1, 1}};

        Eigen::VectorXd inputs {{1, 1, 1}};
        Layer simpleLayer(numOutputs, numInputs, weights, biases);
        simpleLayer.setInputs(inputs);
        simpleLayer.calculate();

        Eigen::Vector2d expected_output {{4, 4}};
        EXPECT_EQ(simpleLayer.getActivations(), expected_output);
    }

    {
        const long numInputs = 3;
        const long numOutputs = 2;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> weights {
                {0, 0, 0},
                {1, 1, 1}
        };
        Eigen::VectorXd biases {{0, 1}};

        Eigen::VectorXd inputs {{-1, -1, -1}};
        Layer simpleLayer(numOutputs, numInputs, weights, biases);
        simpleLayer.setInputs(inputs);
        simpleLayer.calculate();

        Eigen::Vector2d expected_output {{0, 0}};
        EXPECT_EQ(simpleLayer.getActivations(), expected_output);
    }

}

