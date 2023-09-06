//
// Created by Joshua Riefman on 2023-02-20.
//

#include "../include/helpers.h"


int helpers::MaxInArray(std::vector<int>& array) {
    if (array.empty()) {
        throw std::invalid_argument("Trying to find maximum of empty array!");
    }

    int max = 0;

    for (int i : array) {
        if (i > max) { max = i; }
    }

    return max;
}

double helpers::ReLU(double value) {
    return value > 0 ? value : 0;
}

int helpers::Sum(std::vector<int>& array) {
    if (array.empty()) {
        throw std::invalid_argument("Trying to find sum of empty array!");
    }

    int result = 0;

    for (int i : array) {
        result += i;
    }

    return result;
}

double helpers::GetRandomNormalized() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distribution(-1000, 1000);
    return (float)distribution(gen)/1000;
}