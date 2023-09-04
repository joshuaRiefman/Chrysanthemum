//
// Created by Joshua Riefman on 2023-02-20.
//

#include <utility>

#include "../include/Layer.h"
#include "../include/helpers.h"

Layer::Layer(long numOutputs, long numInputs, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& weights, Eigen::VectorXd& biases) {
    this->activations.resize(numOutputs);
    this->inputs.resize(numInputs);
    this->weights = std::move(weights);
    this->biases = std::move(biases);
    this->numInputs = (int)numInputs;
    this->numOutputs = (int)numOutputs;
}

// perform NN calculations using matrix manipulations: weights * inputs + biases
void Layer::evaluate() {
    this->activations = this->weights * this->inputs + this->biases;
    for (int i = 0; i < this->activations.size(); i++) {
        this->activations(i, 0) = helpers::ReLU(this->activations(i, 0));
    }
}
