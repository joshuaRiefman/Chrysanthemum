//
// Created by Joshua Riefman on 2023-02-20.
//

#include "../include/NeuralNetwork.h"
#include <iostream>
#include <utility>

bias_t GetRandomBiases(std::vector<int> layerSizes, int numLayers) {
    bias_t biases;

    const int columns = helpers::MaxInArray(&layerSizes);
    const int rows = numLayers;
    biases.resize(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
//            double randomizedBiases = helpers::GetRandomNormalized();
            double randomizedBiases = 1;

            biases(i, j) = std::make_shared<double>(randomizedBiases);
        }
    }

    return biases;
}

weight_t GetRandomWeights(std::vector<int> layerSizes, const int numLayers, int numInputs) {
    weight_t weights;

    const int columns = helpers::MaxInArray(&layerSizes);
    const int rows = numLayers;
    weights.resize(numLayers, columns);

    for (int i = 0; i < rows; i++) {
        int layerInputCount = i == 0 ? numInputs : layerSizes[i - 1];

        for (int j = 0; j < columns; j++) {
            std::vector<double> randomizedWeights;
            randomizedWeights.resize(layerInputCount);

            for (int k = 0; k < layerInputCount; k++) {
//                randomizedWeights[k] = helpers::GetRandomNormalized();
                randomizedWeights[k] = 1;
            }

            weights(i, j) = std::make_shared<std::vector<double>>(randomizedWeights);

        }
    }

    return weights;
}

void NeuralNetwork::Solve() {
    for (int i = 0; i < layers[0]->inputs.size(); ++i) {
        layers[0]->inputs[i] = inputLayer->network_inputs[i];
    }

    for (int i = 0; i < size; i++) {
        if (i != 0) {
            for (int j = 0; j < layers[i]->inputs.size(); ++j) {
                layers[i]->inputs[j] = layers[i-1]->neurons[j]->activation;
            }
        }

        for (int j = 0; j < layers[i]->neurons.size(); j++) {
            for (int k = 0; k < layers[i]->inputs.size(); k++) {
                layers[i]->neurons[j]->activation += layers[i]->inputs[k] * (*layers[i]->neurons[j]->weights)[k];
            }
            layers[i]->neurons[j]->activation += (*layers[i]->neurons[j]->bias);
            layers[i]->neurons[j]->activation = helpers::ReLU(layers[i]->neurons[j]->activation);
        }
    }
}

int NeuralNetwork::GetHighestNeuronActivationById() {
    double highestActivation = 0;
    int neuronWithHighestActivationID;

    for (int i = 0; i < layers[layers.size() - 1]->neurons.size(); i++) {
        if (layers[layers.size() - 1]->neurons[i]->activation > highestActivation) {
            highestActivation = layers[layers.size() - 1]->neurons[i]->activation;
            neuronWithHighestActivationID = i;
        }
    }
    return neuronWithHighestActivationID;
}

NeuralNetwork::NeuralNetwork(const size_t numNetworkInputs, const std::vector<int> &layerSizes, const weight_t &in_weights, const bias_t &in_biases) {
    int maxNeuronCountPerLayer = *std::max_element(layerSizes.begin(), layerSizes.end());
    size = layerSizes.size();
    layers.resize(size);
    weights.resize((long)size, maxNeuronCountPerLayer);
    biases.resize((long)size, maxNeuronCountPerLayer);

    weights = in_weights;
    biases = in_biases;

    for (int i = 0; i < size; i++) {
        size_t numOutputs = layerSizes[i];
        size_t numInputs = i == 0 ? numNetworkInputs : layerSizes[i-1];

        layers[i] = std::make_unique<HiddenLayer>(HiddenLayer(numOutputs, numInputs));
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < layers[i]->neurons.size(); j++) {
            layers[i]->neurons[j]->weights = weights(i, j);
            layers[i]->neurons[j]->bias = biases(i, j);
        }
    }
//    for (int i = 0; i < size; i++) {
//        printf("Here!1");
//        double value = layers[i]->neurons[i]->activation;
//        layers[i]->outputs[i] = std::make_shared<double>(value);
//    }
}

std::vector<double> NeuralNetwork::GetOutputVector() {
    std::vector<double> output;
    output.resize(layers[size-1]->neurons.size());

    for (int i = 0; i < layers[size-1]->neurons.size(); ++i) {
        output[i] = layers[size-1]->neurons[i]->activation;
    }

    return output;
}

void NeuralNetwork::SetInputs(std::shared_ptr<InputLayer> input) {
    this->inputLayer = std::move(input);
}
