//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_HELPERS_H
#define CHRYSANTHEMUM_HELPERS_H


#include <vector>
#include <random>
#include <chrono>
#include <ctime>

namespace helpers {
    int MaxInArray(std::vector<int>& array);

    double ReLU(double value);

    int Sum(std::vector<int>& array);

    double GetRandomNormalized();
}

#endif //CHRYSANTHEMUM_HELPERS_H
