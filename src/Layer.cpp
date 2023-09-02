//
// Created by Joshua Riefman on 2023-02-20.
//

#include <utility>

#include "../include/Layer.h"

Layer::Layer(long numOutputs, long numInputs, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& weights, Eigen::VectorXd& biases) {
    this->activations.resize(numOutputs);
    this->inputs.resize(numInputs);
    this->weights = std::move(weights);
    this->biases = std::move(biases);
}
