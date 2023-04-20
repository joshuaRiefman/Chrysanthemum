//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_CHRYSANTHEMUM_H
#define CHRYSANTHEMUM_CHRYSANTHEMUM_H

#include <string>
#include <iostream>
#include <fstream>
#include <array>
#include <utility>
#include <vector>
#include <random>
#include <chrono>
#include <ctime>
#include "City.h"
#include "Neuron.h"
#include "Layer.h"
#include "helpers.h"
#include "NeuralNetwork.h"

const std::string DataJSONPath = "../data/data.json";
const std::string ConstantsJSONPath = "../data/constants.json";
const std::string PathJSONPath = "../data/path.json";
const unsigned long universeSize = 15;

std::vector<float> &GetArrayFromJSON(int planetID, const Json::Value *data, std::vector<float> &distances, const std::string &fieldAccessor);

std::vector<City> cities;
int originCityID;

City ParseCityData(int cityID);

void UpdateUniverseConstants();

void SetOrigin();

void InitializePlanets();

static InputLayer SetNetworkInputs(std::vector<int> *planetIDList);

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> GetRandomBiases(std::vector<int> layerSizes, int numLayers);

Eigen::Matrix<std::vector<double>, Eigen::Dynamic, Eigen::Dynamic> GetRandomWeights(std::vector<int> layerSizes, int numLayers, int numInputs);


#endif //CHRYSANTHEMUM_CHRYSANTHEMUM_H
