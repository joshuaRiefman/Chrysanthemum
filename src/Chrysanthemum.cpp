#include "../include/Chrysanthemum.h"

weights_tensor_t Chrysanthemum::getNewWeights(std::vector<int>& layerSizes, int numInputs, ParameterType type) {
    if (numInputs < 1 || layerSizes.empty()) {
        throw std::invalid_argument("Invalid arguments provided to getNewWeights!");
    }

    weights_tensor_t weights;

    for (int i = 0; i < layerSizes.size(); i++) {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> weightMatrix;
        const int rows = layerSizes[i];
        const int columns = i == 0 ? numInputs : layerSizes[i - 1];
        weightMatrix.resize(rows, columns);
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < columns; k++) {
                switch (type) {
                    case RANDOM:
                        weightMatrix(j, k) = helpers::GetRandomNormalized();
                        break;
                    case STANDARD:
                        weightMatrix(j, k) = 1;
                        break;
                    default:
                        throw std::invalid_argument("ParameterType not matched with case!");
                }
            }
        }
        weights.push_back(weightMatrix);
    }

    return weights;
}

biases_matrix_t Chrysanthemum::getNewBiases(std::vector<int>& layerSizes, ParameterType type) {
    if (layerSizes.empty()) {
        throw std::invalid_argument("Empty layerSizes provided to getNewBiases!");
    }

    biases_matrix_t biases;

    for (int numBiases : layerSizes) {
        Eigen::Matrix<double, Eigen::Dynamic, 1> biasVector;
        biasVector.resize(numBiases, 1);
        for (int j = 0; j < numBiases; j++) {
            switch (type) {
                case RANDOM:
                    biasVector[j] = helpers::GetRandomNormalized();
                    break;
                case STANDARD:
                    biasVector[j] = 1;
                    break;
                default:
                    throw std::invalid_argument("ParameterType not matched with case!");
            }
        }
        biases.push_back(biasVector);
    }
    return biases;
}
