//
// Created by Joshua Riefman on 2023-02-20.
//

#include "../include/NeuralNetwork.h"
#include <iostream>
#include <utility>


void NeuralNetwork::solve(std::vector<double> &inputs) {
    if (inputs.size() != this->numInputs) {
        throw InvalidConfiguration("Number of inputs supplied to NN does not match the number designated when the NN was constructed!");
    }

    // load inputs into first layer
    for (int i = 0; i < layers[0]->inputs.size(); i++) {
        layers[0]->inputs[i] = inputs[i];
    }

    // perform NN calculations using matrix manipulations: weights * inputs + biases
    for (int i = 0; i < size; i++) {
        layers[i]->activations = layers[i]->weights * layers[i]->inputs + layers[i]->biases;

        // apply ReLU and propagate activation to the input of the subsequent layer
        for (int j = 0; j < layers[i]->activations.size(); j++) {
            // for the last layer, fill the output vector instead
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
    this->numInputs = numInputs;
    this->weights_tensor = std::move(weights_tensor);
    this->biases_tensor = std::move(biases_tensor);

    for (int i = 0; i < layerSizes.size(); i++) {
        int numLayerOutputs = layerSizes.at(i);
        int numLayerInputs = i == 0 ? numInputs : layerSizes.at(i - 1);
        this->layers.push_back(std::make_unique<Layer>(numLayerOutputs, numLayerInputs, (*this->weights_tensor)[i], (*this->biases_tensor)[i]));
    }

    try {
        verifyConfiguration();
    } catch (InvalidConfiguration& exception) {
        std::cout << exception.what() << std::endl;
        throw;
    }
}

void NeuralNetwork::verifyConfiguration() {
    for (int i = 0; i < size; i++) {
        if (layers[i]->weights.cols() != layers[i]->inputs.size()) {
            throw InvalidConfiguration("Invalid weight column size!");
        }
        if (layers[i]->weights.rows() != layers[i]->activations.size()) {
            throw InvalidConfiguration("Invalid weight row size!");
        }
        if (layers[i]->biases.size() != layers[i]->activations.size()) {
            throw InvalidConfiguration("Invalid biases length!");
        }
    }
}

NeuralNetwork::InvalidConfiguration::InvalidConfiguration(char *message) {
    this->message = message;
}

NeuralNetwork::InvalidConfiguration::InvalidConfiguration(const std::string& message) {
    const int length = (int)message.length();
    char* char_array = new char[length + 1];
    strcpy(char_array, message.c_str());
    this->message = char_array;
}

char* NeuralNetwork::InvalidConfiguration::what() {
    return message;
}
