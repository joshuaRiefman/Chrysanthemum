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

std::vector<float> &GetArrayFromJSON(int planetID, const Json::Value *data, std::vector<float> &distances, const std::string &fieldAccessor);

std::vector<City> cities;
int originCityID;

City ParseCityData(int cityID, int citiesCount);

void UpdateUniverseConstants();

void SetOrigin();

void InitializePlanets(int citiesCount);

static std::shared_ptr<InputLayer> SetNetworkInputs();


#endif //CHRYSANTHEMUM_CHRYSANTHEMUM_H
