//
// Created by Joshua Riefman on 2023-02-20.
//

#include "../include/NeuralNetwork.h"
#include <iostream>
#include <utility>

void NeuralNetwork::solve(std::vector<double> &inputs) {
    // load inputs into first layer
    for (int i = 0; i < layers[0]->inputs.size(); i++) {
        layers[0]->inputs[i] = inputs[i];
    }

    // perform matrix multiplication of weights * inputs + biases
    for (int i = 0; i < size; i++) {
        layers[i]->activations = layers[i]->weights * layers[i]->inputs + layers[i]->biases;

        // apply ReLU and at the last layer, fill the output vector instead of the subsequent layer
        for (int j = 0; j < layers[i]->activations.size(); j++) {
            if (i == size - 1) {
                this->outputs.push_back(helpers::ReLU(layers[i]->activations(j, 0)));
            } else {
                layers[i+1]->inputs(j, 0) = helpers::ReLU(layers[i]->activations(j, 0));
            }
        }
    }
}

NeuralNetwork::NeuralNetwork(const int numInputs, const std::vector<int>& layerSizes, std::unique_ptr<weights_tensor_t>& weights_tensor, std::unique_ptr<biases_matrix_t>& biases_tensor) {
    this->size = layerSizes.size();
    this->weights_tensor = std::move(weights_tensor);
    this->biases_tensor = std::move(biases_tensor);

    for (int i = 0; i < layerSizes.size(); i++) {
        int numLayerOutputs = layerSizes.at(i);
        int numLayerInputs = i == 0 ? numInputs : layerSizes.at(i - 1);
        this->layers.push_back(std::make_unique<Layer>(numLayerOutputs, numLayerInputs, (*this->weights_tensor)[i], (*this->biases_tensor)[i]));
    }
}
