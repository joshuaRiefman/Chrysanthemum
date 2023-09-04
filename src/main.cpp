#include "../include/Chrysanthemum.h"

int main() {
    std::vector<int> layerSizes = {3, 3, 2};
    std::vector<double> inputs = {1, 2, 3};

    std::unique_ptr<weights_tensor_t> weights = std::make_unique<weights_tensor_t>(Chrysanthemum::getNewWeights(layerSizes, (int)inputs.size(), Chrysanthemum::STANDARD));
    std::unique_ptr<biases_matrix_t> biases = std::make_unique<biases_matrix_t>(Chrysanthemum::getNewBiases(layerSizes, Chrysanthemum::STANDARD));
    NeuralNetwork neuralNetwork = NeuralNetwork((int)inputs.size(), layerSizes, weights, biases);
    neuralNetwork.solve(inputs);

    std::vector<double> outputs = neuralNetwork.getOutputs();
    std::cout << "Executed successfully! " <<  std::endl;
}