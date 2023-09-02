//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_HELPERS_H
#define CHRYSANTHEMUM_HELPERS_H

#include <vector>
#include <random>
#include <chrono>
#include <ctime>

class helpers {
public:
    static int MaxInArray(std::vector<int>& array);

    static double ReLU(double value);

    static int Sum(std::vector<int>& array);

    static double GetRandomNormalized();

    static long GetDuration(const std::chrono::time_point<std::chrono::system_clock>& start);
};


#endif //CHRYSANTHEMUM_HELPERS_H
