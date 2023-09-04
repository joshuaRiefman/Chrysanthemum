//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_CHRYSANTHEMUM_H
#define CHRYSANTHEMUM_CHRYSANTHEMUM_H

#include <string>
#include <iostream>
#include <fstream>
#include "NeuralNetwork.h"

// TODO: Maybe should be a namespace? We will never be constructing a Chrysanthemum object.
class Chrysanthemum {
public:
    enum ParameterType {
        STANDARD,
        RANDOM
    };

    static weights_tensor_t getNewWeights(std::vector<int>& layerSizes, int numInputs, ParameterType type = STANDARD);
    static biases_matrix_t getNewBiases(std::vector<int>& layerSizes, ParameterType type = STANDARD);
};

#endif //CHRYSANTHEMUM_CHRYSANTHEMUM_H
