//
// Created by Joshua Riefman on 2023-02-20.
//

#include "Neuron.h"

Neuron::Neuron(double activation, const std::vector<double> &weights, double bias)
        : activation(activation), weights(weights), bias(bias) {}

Neuron::Neuron() : activation(0), weights({1, 1, 1}), bias(1.5) {};

int Neuron::GetHighestNeuronActivationById(std::vector<Neuron> *neurons) {
    double highestActivation = 0;
    int neuronWithHighestActivationID;

    for (int i = 0; i < neurons->size(); i++) {
        if ((*neurons)[i].activation > highestActivation) {
            highestActivation = (*neurons)[i].activation;
            neuronWithHighestActivationID = i;
        }
    }

    return neuronWithHighestActivationID;
}
