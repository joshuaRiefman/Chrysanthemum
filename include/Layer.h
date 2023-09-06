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
    Eigen::VectorXd activations;
    Eigen::VectorXd inputs;
public:
    int numInputs;
    int numOutputs; // number of neurons

    explicit Layer(long numOutputs, long numInputs, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& weights, Eigen::VectorXd& biases);
    void calculate();
    void verifyConfiguration();
    void setInputs(Eigen::VectorXd &new_inputs);
    void setInput(double value, int index);
    Eigen::VectorXd getActivations();
    double getActivation(int index);
};

#endif //CHRYSANTHEMUM_LAYER_H
