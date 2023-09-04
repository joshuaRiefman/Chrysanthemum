//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_LAYER_H
#define CHRYSANTHEMUM_LAYER_H

#include "../external/Eigen/Eigen"

struct Layer {
    //TODO: Make weights, biases, activations readonly with setters/getters
    //TODO: Make matrix sizes template parameters
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> weights;
    Eigen::VectorXd biases;
    Eigen::VectorXd activations;
    Eigen::VectorXd inputs;
    int numInputs;
    int numOutputs; // number of neurons

    explicit Layer(long numOutputs, long numInputs, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& weights, Eigen::VectorXd& biases);
    void evaluate();
};

#endif //CHRYSANTHEMUM_LAYER_H
