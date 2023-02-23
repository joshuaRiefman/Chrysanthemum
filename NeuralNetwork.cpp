//
// Created by Joshua Riefman on 2023-02-20.
//

#include "NeuralNetwork.h"
#include "helpers.h"

NeuralNetworkConfiguration::NeuralNetworkConfiguration(const vector<int> &layerSizes, InputLayer inputs,
                                                       const Matrix<vector<double>, Dynamic, Dynamic> &weights,
                                                       const Matrix<double, Dynamic, Dynamic> &biases,
                                                       const vector<int> &planetIDList, int numOutputs)
                                                       : layerSizes(layerSizes), inputValues(std::move(inputs)), weights(weights), biases(biases), planetIDList(planetIDList), numOutputs(numOutputs) {}

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

NeuralNetwork::NeuralNetwork(NeuralNetwork *network, NeuralNetworkConfiguration *config) {
    int maxNeuronCountPerLayer = helpers::MaxInArray(&config->layerSizes);
    network->size = (int)config->layerSizes.size();
    network->layers.resize(network->size);
    network->weights.resize(network->size, maxNeuronCountPerLayer);
    network->biases.resize(network->size, maxNeuronCountPerLayer);

    network->weights = config->weights;
    network->biases = config->biases;

    network->inputLayer = config->inputValues;

    for (int i = 0; i <= network->size; i++) {
        int numOutputs = config->layerSizes[i];
        int numInputs;
        if (i-1 < 0) {
            numInputs = (int)config->inputValues.inputs.size();
        } else
        {
            numInputs = config->layerSizes[i-1];
        }

        network->layers[i] = Layer::CreateLayer(numOutputs, numInputs);
    }

    for (int i = 0; i < network->size; i++) {
        for (int j = 0; j < maxNeuronCountPerLayer; j++) {
            network->layers[i].outputs[j].weights = network->weights(i, j);
            network->layers[i].outputs[j].bias = network->biases(i, j);
        }
    }
}

void NeuralNetwork::Solve(NeuralNetwork *network) {
    for (int i = 0; i < network->layers[0].inputs.size(); ++i) {
        network->layers[0].inputs[i] = network->inputLayer.outputs[i].activation;
    }

    for (int i = 0; i < network->size; i++) {
        Layer *layer = &network->layers[i];
        if (i != 0) {
            for (int j = 0; j < layer->inputs.size(); ++j) {
                layer->inputs[j] = network->layers[i-1].outputs[j].activation;
            }
        }

        for (int j = 0; j < layer->outputs.size(); j++) {
            for (int k = 0; k < layer->inputs.size(); k++) {
                layer->outputs[j].activation += layer->inputs[k] * layer->outputs[j].weights[k];
            }
            layer->outputs[j].activation += layer->outputs[j].bias;
            layer->outputs[j].activation = helpers::ReLU(layer->outputs[j].activation);
        }
    }
}

NeuralNetwork::NeuralNetwork() : layers(), size() {}
