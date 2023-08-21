//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_LAYER_H
#define CHRYSANTHEMUM_LAYER_H

#include "Neuron.h"

struct Layer {
    std::vector<std::unique_ptr<Neuron>> neurons;
};

struct HiddenLayer : Layer {
    std::vector<double> inputs;

    explicit HiddenLayer(size_t numOutputs, size_t numInputs);
};

struct InputLayer : Layer {
    std::vector<double> network_inputs;
    size_t numInputs;

    explicit InputLayer(const std::vector<double>&inputs);
};


#endif //CHRYSANTHEMUM_LAYER_H
