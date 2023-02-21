//
// Created by Joshua Riefman on 2023-02-20.
//

#include "NeuralNetwork.h"
#include "helpers.h"

NeuralNetworkConfiguration::NeuralNetworkConfiguration(const vector<int> &layerSizes, InputLayer inputs,
                           const Matrix<vector<double>, Dynamic, Dynamic> &weights,
                           const Matrix<double, Dynamic, Dynamic> &biases, const vector<int> &planetIDList)
        : layerSizes(layerSizes), inputValues(std::move(inputs)), weights(weights), biases(biases), planetIDList(planetIDList) {}


