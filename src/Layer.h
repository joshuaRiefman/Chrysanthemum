//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_LAYER_H
#define CHRYSANTHEMUM_LAYER_H

#include "Neuron.h"

class Layer {
public:
    vector<Neuron> outputs;
    vector<double> inputs;

    Layer(int numOutputs, int numInputs);

    Layer();

    static Layer CreateLayer(int numOutputs, int numInputs);
};

struct InputLayer : Layer {

    vector<double> d_inputs;

    explicit InputLayer(vector<double> *INPUTS);

    InputLayer();
};


#endif //CHRYSANTHEMUM_LAYER_H
