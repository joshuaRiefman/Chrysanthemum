//
// Created by Joshua Riefman on 2023-02-20.
//

#include "Layer.h"


Layer::Layer(int numOutputs, int numInputs) {
    for (int i = 0; i < numOutputs; ++i) {
        outputs.emplace_back();
    }

    for (int i = 0; i < numInputs; ++i) {
        inputs.emplace_back();
    }
}


Layer::Layer() = default;

Layer Layer::CreateLayer(int numOutputs, int numInputs) {
    return {numOutputs, numInputs};
}

InputLayer::InputLayer(std::vector<double> *INPUTS) {
    d_inputs = *INPUTS;

    outputs.resize(d_inputs.size());

    for (int i = 0; i < outputs.size(); ++i) {
        outputs[i].activation = d_inputs[i];
    }
}

InputLayer::InputLayer() = default;
