//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_NEURALNETWORK_H
#define CHRYSANTHEMUM_NEURALNETWORK_H

#include "Neuron.h"
#include "Layer.h"
#include "helpers.h"
#include "../include/Eigen/Eigen"

using Eigen::Dynamic;
using Eigen::Matrix;

struct NeuralNetworkConfiguration {
    vector<int> layerSizes;
    vector<int> planetIDList;
    InputLayer inputValues;
    Matrix<vector<double>, Dynamic, Dynamic> weights; //TODO: Convert to Tensor!
    Matrix<double, Dynamic, Dynamic> biases;

    NeuralNetworkConfiguration(const vector<int> &layerSizes,
                               InputLayer inputs,
                               const Matrix<vector<double>,Dynamic, Dynamic> &weights,
                               const Matrix<double, Dynamic, Dynamic> &biases,
                               const vector<int> &planetIDList);
};

class NeuralNetwork {
public:
    NeuralNetwork();

    Matrix<vector<double>, Dynamic, Dynamic> weights;
    Matrix<double, Dynamic, Dynamic> biases;
    vector<Layer> layers;
    InputLayer inputLayer;
    int size;

    NeuralNetwork(NeuralNetwork *network, std::unique_ptr<NeuralNetworkConfiguration> config);

    static void Solve(NeuralNetwork *network);
    static int GetHighestNeuronActivationById(std::unique_ptr<vector<Neuron>> neurons);
};

Matrix<vector<double>, Dynamic, Dynamic> GetRandomWeights(vector<int> layerSizes, int numLayers, int numInputs);

Matrix<double, Dynamic, Dynamic> GetRandomBiases(vector<int> layerSizes, int numLayers);

#endif //CHRYSANTHEMUM_NEURALNETWORK_H
