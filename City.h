//
// Created by Joshua Riefman on 2023-02-21.
//

#ifndef CHRYSANTHEMUM_CITY_H
#define CHRYSANTHEMUM_CITY_H

#include "vector"

class City {
public:
    int id{};
    double distanceFromOrigin{};
    std::vector<float> distances;
    std::vector<float> deltaDistances;
    bool visited{};

    City();

    City(int ID, std::vector<float>& DISTANCES, double DISTANCE_FROM_ORIGIN, std::vector<float>& DELTA_DISTANCES, bool VISITED);
};


#endif //CHRYSANTHEMUM_CITY_H
