//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_NEURON_H
#define CHRYSANTHEMUM_NEURON_H

#include <vector>

using std::vector;

class Neuron {
public:
    double bias{};
    double activation{};
    vector<double> weights{};

    Neuron(double activation, const vector<double> &weights, double bias);

    Neuron();

    static int GetHighestNeuronActivationById(vector<Neuron> *neurons);
};


#endif //CHRYSANTHEMUM_NEURON_H
