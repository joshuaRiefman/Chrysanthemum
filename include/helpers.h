//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_HELPERS_H
#define CHRYSANTHEMUM_HELPERS_H


#include <vector>
#include <random>
#include <chrono>
#include <ctime>
// TODO: Make this into a namespace!
class helpers {
public:
    // TODO: Make this generic with a constraint that the type is numeric!
    static int MaxInArray(std::vector<int>& array);

    static double ReLU(double value);

    static int Sum(std::vector<int>& array);

    static double GetRandomNormalized();

    static long GetDuration(const std::chrono::time_point<std::chrono::system_clock>& start);
};

#endif //CHRYSANTHEMUM_HELPERS_H
