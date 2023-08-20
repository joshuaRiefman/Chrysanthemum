//
// Created by Joshua Riefman on 2023-02-20.
//

#include "../include/NeuralNetwork.h"
#include "../include/helpers.h"

NeuralNetworkConfiguration::NeuralNetworkConfiguration(const vector<int> &layerSizes, InputLayer inputs,
                                                       const Matrix<vector<double>, Dynamic, Dynamic> &weights,
                                                       const Matrix<double, Dynamic, Dynamic> &biases,
                                                       const vector<int> &planetIDList)
                                                       : layerSizes(layerSizes), inputValues(std::move(inputs)), weights(weights), biases(biases), planetIDList(planetIDList) {}

Matrix<double, Dynamic, Dynamic> GetRandomBiases(vector<int> layerSizes, int numLayers) {
    Matrix<double, Dynamic, Dynamic> biases;

    const int columns = helpers::MaxInArray(&layerSizes);
    const int rows = numLayers;
    biases.resize(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
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

    for (int i = 0; i < rows; i++) {
        int layerInputCount = i == 0 ? numInputs : layerSizes[i - 1];

        for (int j = 0; j < columns; j++) {
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

NeuralNetwork::NeuralNetwork(NeuralNetwork *network, std::unique_ptr<NeuralNetworkConfiguration> config) {
    int maxNeuronCountPerLayer = helpers::MaxInArray(&config->layerSizes);
    network->size = (int)config->layerSizes.size();
    network->layers.resize(network->size);
    network->weights.resize(network->size, maxNeuronCountPerLayer);
    network->biases.resize(network->size, maxNeuronCountPerLayer);

    network->weights = config->weights;
    network->biases = config->biases;

    network->inputLayer = config->inputValues;

    for (int i = 0; i < network->size; i++) {
        int numOutputs = config->layerSizes[i];
        int numInputs = i == 0 ? (int)config->inputValues.d_inputs.size() : config->layerSizes[i-1];

        network->layers[i] = make_unique<Layer>(Layer::CreateLayer(numOutputs, numInputs));
    }

    for (int i = 0; i < network->size; i++) {
        for (int j = 0; j < network->layers[i]->outputs.size(); j++) {
            network->layers[i]->outputs[j].weights = network->weights(i, j);
            network->layers[i]->outputs[j].bias = network->biases(i, j);
        }
    }
}

void NeuralNetwork::Solve(NeuralNetwork *network) {
    for (int i = 0; i < network->layers[0]->inputs.size(); ++i) {
        network->layers[0]->inputs[i] = network->inputLayer.outputs[i].activation;
    }

    for (int i = 0; i < network->size; i++) {
        if (i != 0) {
            for (int j = 0; j < network->layers[i]->inputs.size(); ++j) {
                network->layers[i]->inputs[j] = network->layers[i-1]->outputs[j].activation;
            }
        }

        for (int j = 0; j < network->layers[i]->outputs.size(); j++) {
            for (int k = 0; k < network->layers[i]->inputs.size(); k++) {
                network->layers[i]->outputs[j].activation += network->layers[i]->inputs[k] * network->layers[i]->outputs[j].weights[k];
            }
            network->layers[i]->outputs[j].activation += network->layers[i]->outputs[j].bias;
            network->layers[i]->outputs[j].activation = helpers::ReLU(network->layers[i]->outputs[j].activation);
        }
    }
}

int NeuralNetwork::GetHighestNeuronActivationById(std::unique_ptr<vector<Neuron>> neurons) {
    double highestActivation = 0;
    int neuronWithHighestActivationID;

    for (int i = 0; i < neurons->size(); i++) {
        if ((*neurons)[i].activation > highestActivation) {
            highestActivation = (*neurons)[i].activation;
            neuronWithHighestActivationID = i;
        }
    }

    return neuronWithHighestActivationID;
}

NeuralNetwork::NeuralNetwork() : layers(), size() {}
