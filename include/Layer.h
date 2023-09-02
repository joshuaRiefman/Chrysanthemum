//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_LAYER_H
#define CHRYSANTHEMUM_LAYER_H

#include "../external/Eigen/Eigen"

struct Layer {
    Eigen::VectorXd activations;

    Eigen::VectorXd inputs;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> weights;
    Eigen::VectorXd biases;

    explicit Layer(long numOutputs, long numInputs, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& weights, Eigen::VectorXd& biases);

};

#endif //CHRYSANTHEMUM_LAYER_H
