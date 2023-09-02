//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_NEURALNETWORK_H
#define CHRYSANTHEMUM_NEURALNETWORK_H

#include "Layer.h"
#include "helpers.h"

typedef std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> weights_tensor_t;
typedef std::vector<Eigen::VectorXd> biases_matrix_t;

class NeuralNetwork {
    size_t size; // number of layers
    std::unique_ptr<weights_tensor_t> weights_tensor;
    std::unique_ptr<biases_matrix_t> biases_tensor;
    std::vector<std::unique_ptr<Layer>> layers;
public:
    std::vector<double> outputs;
    explicit NeuralNetwork(int numInputs, const std::vector<int>& layerSizes, std::unique_ptr<weights_tensor_t>& weights_tensor, std::unique_ptr<biases_matrix_t>& biases_tensor);
    void solve(std::vector<double> &inputs);
};


#endif //CHRYSANTHEMUM_NEURALNETWORK_H
