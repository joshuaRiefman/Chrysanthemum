//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_NEURALNETWORK_H
#define CHRYSANTHEMUM_NEURALNETWORK_H

#include "Neuron.h"
#include "Layer.h"
#include "helpers.h"
#include "Packages/eigen-3.4.0/Eigen/Eigen"

using namespace Eigen;

struct NeuralNetworkConfiguration {
    vector<int> layerSizes;
    vector<int> planetIDList;
    InputLayer inputValues;

    NeuralNetworkConfiguration(const vector<int> &layerSizes, InputLayer inputs,
                               const Matrix<vector<double>, Dynamic, Dynamic> &weights,
                               const Matrix<double, Dynamic, Dynamic> &biases, const vector<int> &planetIDList);
    Matrix<vector<double>, Dynamic, Dynamic> weights;
    Matrix<double, Dynamic, Dynamic> biases;
};

template<int networkSize>
class NeuralNetwork {
public:
    NeuralNetwork();

    Matrix<vector<double>, Dynamic, Dynamic> weights;
    Matrix<double, Dynamic, Dynamic> biases;
    Layer layers[networkSize];
    InputLayer inputLayer;
    int size;

    NeuralNetwork(NeuralNetwork<networkSize> *network, NeuralNetworkConfiguration *config);

    void Solve(NeuralNetwork<networkSize> *network);
};

template<int networkSize>
NeuralNetwork<networkSize>::NeuralNetwork(NeuralNetwork<networkSize> *network, NeuralNetworkConfiguration *config) {
    int maxNeuronCountPerLayer = helpers::MaxInArray(&config->layerSizes);
    network->size = networkSize;
    network->weights.resize(network->size, maxNeuronCountPerLayer);
    network->biases.resize(network->size, maxNeuronCountPerLayer);

    network->weights = config->weights;
    network->biases = config->biases;

    network->inputLayer = config->inputValues;

    for (int i = 0; i < networkSize; i++) {
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

template<int networkSize>
void NeuralNetwork<networkSize>::Solve(NeuralNetwork<networkSize> *network) {
    for (int i = 0; i < network->layers[0].inputs.size(); ++i) {
        network->layers[0].inputs[i] = network->inputLayer.outputs[i].activation;
    }

    for (int i = 0; i < networkSize; i++) {
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

template<int networkSize>
NeuralNetwork<networkSize>::NeuralNetwork() : layers(), size(networkSize) {}

Matrix<vector<double>, Dynamic, Dynamic> GetRandomWeights(vector<int> layerSizes, int numLayers, int numInputs);

Matrix<double, Dynamic, Dynamic> GetRandomBiases(vector<int> layerSizes, int numLayers);

#endif //CHRYSANTHEMUM_NEURALNETWORK_H
