//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_NEURALNETWORK_H
#define CHRYSANTHEMUM_NEURALNETWORK_H

#include "Layer.h"
#include "helpers.h"
#include "exceptions.h"

typedef std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> weights_tensor_t;
typedef std::vector<Eigen::VectorXd> biases_matrix_t;

class NeuralNetwork {
private:
    bool outputsAreValid;
    size_t size; // number of layers
    size_t numInputs; // number of inputs to the network
    std::vector<double> outputs;
    std::unique_ptr<weights_tensor_t> weights_tensor;
    std::unique_ptr<biases_matrix_t> biases_tensor;
    std::vector<std::unique_ptr<Layer>> layers;
public:
    void verifyConfiguration();
    explicit NeuralNetwork(int numInputs, const std::vector<int>& layerSizes, std::unique_ptr<weights_tensor_t>& weights_tensor, std::unique_ptr<biases_matrix_t>& biases_tensor);
    std::vector<double> solve(const std::vector<double> &inputs);
    std::vector<double> getOutputs();
};


#endif //CHRYSANTHEMUM_NEURALNETWORK_H
