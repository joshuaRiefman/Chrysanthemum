//
// Created by Joshua Riefman on 2023-02-20.
//

#include "../include/Layer.h"

HiddenLayer::HiddenLayer(size_t numOutputs, size_t numInputs) {
    for (int i = 0; i < numOutputs; ++i) {
        this->neurons.emplace_back(std::make_unique<Neuron>());
    }

    for (int i = 0; i < numInputs; ++i) {
        this->inputs.emplace_back(0);
    }
}

InputLayer::InputLayer(const std::vector<double> &inputs) {
    this->network_inputs = inputs;
    this->numInputs = inputs.size();
}