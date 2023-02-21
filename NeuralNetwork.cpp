//
// Created by Joshua Riefman on 2023-02-20.
//

#include "NeuralNetwork.h"
#include "helpers.h"

NeuralNetworkConfiguration::NeuralNetworkConfiguration(const vector<int> &layerSizes, InputLayer inputs,
                           const Matrix<vector<double>, Dynamic, Dynamic> &weights,
                           const Matrix<double, Dynamic, Dynamic> &biases, const vector<int> &planetIDList)
        : layerSizes(layerSizes), inputValues(std::move(inputs)), weights(weights), biases(biases), planetIDList(planetIDList) {}

Matrix<double, Dynamic, Dynamic> GetRandomBiases(vector<int> layerSizes, int numLayers) {
    Matrix<double, Dynamic, Dynamic> biases;

    const int columns = helpers::MaxInArray(&layerSizes);
    const int rows = numLayers;
    biases.resize(rows, columns);

    for (int i = 0; i < columns; i++) {
        for (int j = 0; j < rows; j++) {
            double randomizedBiases = helpers::GetRandomNormalized();

            biases(i, j) = randomizedBiases;
        }
    }

    return biases;
}

Matrix<vector<double>, Dynamic, Dynamic> GetRandomWeights(vector<int> layerSizes, int numLayers, int numInputs) {
    Matrix<vector<double>, Dynamic, Dynamic> weights;

    const int columns = helpers::MaxInArray(&layerSizes);
    const int rows = numLayers;
    weights.resize(numLayers, columns);

    for (int i = 0; i < columns; i++) {
        int layerInputCount = i - 1 < 0 ? numInputs : layerSizes[i - 1];

        for (int j = 0; j < rows; j++) {
            vector<double> randomizedWeights;
            randomizedWeights.resize(layerInputCount);

            for (int k = 0; k < layerInputCount; k++) {
                randomizedWeights[k] = helpers::GetRandomNormalized();
            }

            weights(i, j) = randomizedWeights;
        }
    }

    return weights;
}
