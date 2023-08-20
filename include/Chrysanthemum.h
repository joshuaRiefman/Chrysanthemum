//
// Created by Joshua Riefman on 2023-02-20.
//

#ifndef CHRYSANTHEMUM_CHRYSANTHEMUM_H
#define CHRYSANTHEMUM_CHRYSANTHEMUM_H

#include <string>
#include <iostream>
#include <fstream>
#include "City.h"
#include "NeuralNetwork.h"
#include "../external/jsoncpp/json/json.h"

const std::string DataJSONPath = "../data/data.json";
const std::string ConstantsJSONPath = "../data/constants.json";
const std::string PathJSONPath = "../data/path.json";
const unsigned long universeSize = 15;

vector<float> &GetArrayFromJSON(int planetID, const Json::Value *data, vector<float> &distances, const std::string &fieldAccessor);

vector<City> cities;
int originCityID;

City ParseCityData(int cityID);

void UpdateUniverseConstants();

void SetOrigin();

void InitializePlanets();

static InputLayer SetNetworkInputs(std::vector<int> *planetIDList);

Matrix<double, Dynamic, Dynamic> GetRandomBiases(vector<int> layerSizes, int numLayers);

Matrix<vector<double>, Dynamic, Dynamic> GetRandomWeights(vector<int> layerSizes, int numLayers, int numInputs);


#endif //CHRYSANTHEMUM_CHRYSANTHEMUM_H
