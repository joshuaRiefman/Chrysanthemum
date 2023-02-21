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

using namespace std;
using namespace Eigen;

const string DataJSONPath = "../config/data.json";
const string ConstantsJSONPath = "../config/constants.json";
const string PathJSONPath = "../config/path.json";
const unsigned long universeSize = 15;

vector<float> &GetArrayFromJSON(int planetID, const Json::Value *data, vector<float> &distances, const string &fieldAccessor);

vector<City> cities;
int originCityID;

City ParseCityData(int cityID);

void UpdateUniverseConstants();

void SetOrigin();

void InitializePlanets();

static InputLayer SetNetworkInputs(vector<int> *planetIDList);

Matrix<double, Dynamic, Dynamic> GetRandomBiases(vector<int> layerSizes, int numLayers);

Matrix<vector<double>, Dynamic, Dynamic> GetRandomWeights(vector<int> layerSizes, int numLayers, int numInputs);


#endif //CHRYSANTHEMUM_CHRYSANTHEMUM_H
