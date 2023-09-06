//
// Created by Joshua Riefman on 2023-02-20.
//

#include "../include/NeuralNetwork.h"
#include <iostream>
#include <utility>


std::vector<double> NeuralNetwork::solve(const std::vector<double> &inputs) {
    if (inputs.size() != this->numInputs) {
        throw ChrysanthemumExceptions::InvalidConfiguration("Number of inputs supplied to NN does not match the number designated when the NN was constructed!");
    }

    // load inputs into first layer
    for (int i = 0; i < layers[0]->numInputs; i++) {
        layers[0]->setInput(inputs[i], i);
    }

    // perform NN calculations using matrix manipulations: weights * inputs + biases
    for (int i = 0; i < size; i++) {
        layers[i]->calculate();

        // apply ReLU and propagate activation to the input of the subsequent layer
        for (int j = 0; j < layers[i]->numOutputs; j++) {
            // for the last layer, fill the output vector instead
            if (i == size - 1) {
                this->outputs.push_back(layers[i]->getActivation(j));
            } else {
                layers[i+1]->setInput(layers[i]->getActivation(j), j);
            }
        }
    }

    outputsAreValid = true;
    return outputs;
}

NeuralNetwork::NeuralNetwork(const int numInputs, const std::vector<int>& layerSizes, std::unique_ptr<weights_tensor_t>& weights_tensor, std::unique_ptr<biases_matrix_t>& biases_tensor) {
    this->size = layerSizes.size();
    this->numInputs = numInputs;
    this->weights_tensor = std::move(weights_tensor);
    this->biases_tensor = std::move(biases_tensor);
    this->outputsAreValid = false;

    for (int i = 0; i < layerSizes.size(); i++) {
        int numLayerOutputs = layerSizes.at(i);
        int numLayerInputs = i == 0 ? numInputs : layerSizes.at(i - 1);
        this->layers.push_back(std::make_unique<Layer>(numLayerOutputs, numLayerInputs, (*this->weights_tensor)[i], (*this->biases_tensor)[i]));
    }

    try {
        this->verifyConfiguration();
    } catch (ChrysanthemumExceptions::InvalidConfiguration& exception) {
        std::cout << exception.what() << std::endl;
        throw;
    }
}

void NeuralNetwork::verifyConfiguration() {
    for (const std::unique_ptr<Layer>& layer: layers) {
        layer->verifyConfiguration();
    }
}

std::vector<double> NeuralNetwork::getOutputs() {
    if (outputsAreValid) {
        return this->outputs;
    } else {
        throw ChrysanthemumExceptions::PrematureAccess("Trying to extract outputs of an uncomputed neural network!");
    }
}
