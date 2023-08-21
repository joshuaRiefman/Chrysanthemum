//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_NEURON_H
#define CHRYSANTHEMUM_NEURON_H

#include <vector>

struct Neuron {
    std::shared_ptr<double> bias{};
    double activation{};
    std::shared_ptr<std::vector<double>> weights{};

    explicit Neuron(std::shared_ptr<std::vector<double>> &&weights, std::shared_ptr<double> &bias);

    explicit Neuron();
};


#endif //CHRYSANTHEMUM_NEURON_H
