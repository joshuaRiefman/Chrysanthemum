#include "../include/Chrysanthemum.h"

weights_tensor_t Chrysanthemum::getNewWeights(std::vector<int>& layerSizes, int numInputs, ParameterType type) {
    weights_tensor_t weights;

    for (int i = 0; i < layerSizes.size(); i++) {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> weightMatrix;
        const int rows = layerSizes[i];
        const int columns = i == 0 ? numInputs : layerSizes[i - 1];
        weightMatrix.resize(rows, columns);
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < columns; k++) {
                if (type == RANDOM) {
                    weightMatrix(j, k) = helpers::GetRandomNormalized();
                } else if (type == STANDARD) {
                    weightMatrix(j, k) = 1;
                } else {
                    std::cerr << "ParameterType not supplied!" << std::endl;
                    exit(1);
                }
            }
        }
        weights.push_back(weightMatrix);
    }

    return weights;
}

biases_matrix_t Chrysanthemum::getNewBiases(std::vector<int>& layerSizes, ParameterType type) {
    biases_matrix_t biases;

    for (int numBiases : layerSizes) {
        Eigen::Matrix<double, Eigen::Dynamic, 1> biasVector;
        biasVector.resize(numBiases, 1);
        for (int j = 0; j < numBiases; j++) {
            if (type == RANDOM) {
                biasVector[j] = helpers::GetRandomNormalized();
            } else if (type == STANDARD) {
                biasVector[j] = 1;
            } else {
                std::cerr << "ParameterType not supplied!" << std::endl;
                exit(1);
            }
        }
        biases.push_back(biasVector);
    }
    return biases;
}
