//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_CHRYSANTHEMUM_H
#define CHRYSANTHEMUM_CHRYSANTHEMUM_H

#include <string>
#include <iostream>
#include <fstream>
#include "City.h"
#include "NeuralNetwork.h"
#include "../external/jsoncpp/json/json.h"

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
