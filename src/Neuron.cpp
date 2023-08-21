//
// Created by Joshua Riefman on 2023-02-20.
//

#include "../include/Neuron.h"

Neuron::Neuron() : activation(0), weights(std::make_shared<std::vector<double>>()), bias(std::make_shared<double>(1.5)) {}

Neuron::Neuron(std::shared_ptr<std::vector<double>> &&new_weights, std::shared_ptr<double> &new_bias) {
    activation = 0;

    weights = std::move(new_weights);
    bias = std::move(new_bias);
}
