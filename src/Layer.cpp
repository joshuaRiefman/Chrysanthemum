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
void Layer::calculate() {
    this->activations = this->weights * this->inputs + this->biases;
    for (int i = 0; i < this->activations.size(); i++) {
        this->activations(i, 0) = helpers::ReLU(this->activations(i, 0));
    }
}

void Layer::verifyConfiguration() {
    if (activations.size() != numOutputs) {
        throw ChrysanthemumExceptions::InvalidConfiguration("Wrong number of neurons!");
    }
    if (inputs.size() != numInputs) {
        throw ChrysanthemumExceptions::InvalidConfiguration("Wrong size of inputs!");
    }
    if (weights.cols() != numInputs) {
        throw ChrysanthemumExceptions::InvalidConfiguration("Invalid weight column size!");
    }
    if (weights.rows() != numOutputs) {
        throw ChrysanthemumExceptions::InvalidConfiguration("Invalid weight row size!");
    }
    if (biases.size() != numOutputs) {
        throw ChrysanthemumExceptions::InvalidConfiguration("Invalid biases length!");
    }
}

Eigen::VectorXd Layer::getActivations() {
    return activations;
}

void Layer::setInputs(Eigen::VectorXd &new_inputs) {
    this->inputs = new_inputs;
}

void Layer::setInput(double value, int index) {
    inputs(index, 0) = value;
}

double Layer::getActivation(int index) {
    return activations(index, 0);
}
