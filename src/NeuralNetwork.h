//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_NEURALNETWORK_H
#define CHRYSANTHEMUM_NEURALNETWORK_H

#include "Neuron.h"
#include "Layer.h"
#include "helpers.h"
#include "../include/Eigen/Eigen"

struct NeuralNetworkConfiguration {
    std::vector<int> layerSizes;
    std::vector<int> planetIDList;
    InputLayer inputValues;
    Eigen::Matrix<std::vector<double>, Eigen::Dynamic, Eigen::Dynamic> weights; //TODO: Convert to Tensor!
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> biases;
    int numOutputs;

    NeuralNetworkConfiguration(const std::vector<int> &layerSizes,
                               InputLayer inputs,
                               const Eigen::Matrix<std::vector<double>,Eigen::Dynamic, Eigen::Dynamic> &weights,
                               const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &biases,
                               const std::vector<int> &planetIDList, int numOutputs);
};

class NeuralNetwork {
public:
    NeuralNetwork();

    Eigen::Matrix<std::vector<double>, Eigen::Dynamic, Eigen::Dynamic> weights;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> biases;
    std::vector<Layer> layers;
    InputLayer inputLayer;
    int size;

    NeuralNetwork(NeuralNetwork *network, NeuralNetworkConfiguration *config);

    static void Solve(NeuralNetwork *network);
};

Eigen::Matrix<std::vector<double>, Eigen::Dynamic, Eigen::Dynamic> GetRandomWeights(std::vector<int> layerSizes, int numLayers, int numInputs);

Eigen::Matrix<double,Eigen:: Dynamic, Eigen::Dynamic> GetRandomBiases(std::vector<int> layerSizes, int numLayers);

#endif //CHRYSANTHEMUM_NEURALNETWORK_H
