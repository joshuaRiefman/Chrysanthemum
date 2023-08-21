//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_NEURALNETWORK_H
#define CHRYSANTHEMUM_NEURALNETWORK_H

#include "Neuron.h"
#include "Layer.h"
#include "helpers.h"
#include "../external/Eigen/Eigen"

using weight_t = Eigen::Matrix<std::shared_ptr<std::vector<double>>, Eigen::Dynamic, Eigen::Dynamic>;
using bias_t = Eigen::Matrix<std::shared_ptr<double>, Eigen::Dynamic, Eigen::Dynamic>;

struct NeuralNetwork {
    weight_t weights;
    bias_t biases;
    std::vector<std::unique_ptr<HiddenLayer>> layers;
    std::shared_ptr<InputLayer> inputLayer;
    size_t size;

    explicit NeuralNetwork(size_t numNetworkInputs, const std::vector<int> &layerSizes, const weight_t &in_weights, const bias_t &in_biases);

    void Solve();
    void SetInputs(std::shared_ptr<InputLayer> input);
    int GetHighestNeuronActivationById();
    std::vector<double> GetOutputVector();
};

weight_t GetRandomWeights(std::vector<int> layerSizes, int numLayers, int numInputs);

bias_t GetRandomBiases(std::vector<int> layerSizes, int numLayers);



#endif //CHRYSANTHEMUM_NEURALNETWORK_H
