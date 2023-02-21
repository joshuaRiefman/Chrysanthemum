//
// Created by Joshua Riefman on 2023-02-21.
//

#include "City.h"

City::City() = default;

City::City(int ID, std::vector<float>& DISTANCES, double DISTANCE_FROM_ORIGIN, std::vector<float>& DELTA_DISTANCES, bool VISITED) {
    id = ID;
    distances = DISTANCES;
    distanceFromOrigin = DISTANCE_FROM_ORIGIN;
    deltaDistances = DELTA_DISTANCES;
    visited = VISITED;
}