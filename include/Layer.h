//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_LAYER_H
#define CHRYSANTHEMUM_LAYER_H

#include "../external/Eigen/Eigen"
#include "exceptions.h"

class Layer {
private:
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> weights;
    Eigen::VectorXd biases;
public:
    //TODO: Make activations readonly with setters/getters
    //TODO: Make matrix sizes template parameters
    Eigen::VectorXd activations;
    Eigen::VectorXd inputs;
    // TODO: Make these readonly!
    int numInputs;
    int numOutputs; // number of neurons

    explicit Layer(long numOutputs, long numInputs, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& weights, Eigen::VectorXd& biases);
    void evaluate();
    void verifyConfiguration();
};

#endif //CHRYSANTHEMUM_LAYER_H
