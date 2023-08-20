//
// Created by Joshua Riefman on 2023-02-20.
//

#include "../include/Neuron.h"

Neuron::Neuron(double activation, const vector<double> &weights, double bias)
        : activation(activation), weights(weights), bias(bias) {}

Neuron::Neuron() : activation(0), weights({1, 1, 1}), bias(1.5) {};
