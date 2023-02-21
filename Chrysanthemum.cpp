#include "/Users/joshuariefman/vcpkg/installed/arm64-osx/include/json/value.h"
#include "/Users/joshuariefman/vcpkg/installed/arm64-osx/include/json/reader.h"
#include "/Users/joshuariefman/vcpkg/installed/arm64-osx/include/json/writer.h"
#include "/Users/joshuariefman/vcpkg/installed/arm64-osx/include/json/json.h"
#include "Packages/eigen-3.4.0/Eigen/Eigen"
#include "Neuron.h"
#include "Layer.h"
#include "helpers.h"
#include "NeuralNetwork.h"
#include "Chrysanthemum.h"

using namespace std;
using namespace Eigen;


array<float, universeSize> &GetArrayFromJSON(int planetID, const Json::Value *data, array<float, universeSize> &distances, const string &fieldAccessor) {
    for (int i = 0; i < universeSize; ++i) {
        distances[i] = (*data)["Planets"][planetID][fieldAccessor][i].asFloat();
    }
    return distances;
}

City::City() = default;

City::City(int ID, const array<float, universeSize>& DISTANCES, double DISTANCE_FROM_ORIGIN, const array<float, universeSize>& DELTA_DISTANCES, bool VISITED) {
    id = ID;
    distances = DISTANCES;
    distanceFromOrigin = DISTANCE_FROM_ORIGIN;
    deltaDistances = DELTA_DISTANCES;
    visited = VISITED;
}

City ParseCityData(int cityID) {
    ifstream filePath(DataJSONPath);
    Json::Reader reader;
    Json::Value data;

    reader.parse(filePath, data);
    filePath.close();

    int id = cityID;
    double distanceFromOrigin = data["distance_from_origin"].asDouble();
    array<float, universeSize> distances = GetArrayFromJSON(cityID, &data, distances, "distances");
    array<float, universeSize> deltaDistances = GetArrayFromJSON(cityID, &data, deltaDistances, "deltaDistances");

    City newCity(id, distances, distanceFromOrigin, deltaDistances, false);
    return newCity;
}

void UpdateUniverseConstants() {
    Json::StreamWriterBuilder builder;
    builder["commentStyle"] = "None";
    builder["indentation"] = "    ";

    Json::Value value;
    value["universeSize"] = (int)universeSize;

    ofstream outputFileStream(ConstantsJSONPath);
    builder.newStreamWriter()->write(value, &outputFileStream);

    outputFileStream.close();
}

void SetOrigin() {
    ifstream filePath(DataJSONPath);
    Json::Reader reader;
    Json::Value data;

    reader.parse(filePath, data);
    filePath.close();

    originCityID = data["starting_position"].asInt();
    cities[originCityID].visited = true;
}

void InitializePlanets() {
    for (int i = 0; i < universeSize; i++)
    {
        cities[i] = ParseCityData(i);
    }

    SetOrigin();
}

static InputLayer SetNetworkInputs(vector<int> *planetIDList) {
    vector<double> outputs;
    for (int i = 0; i < universeSize; ++i) {

        if (cities[i].visited) {
            outputs.emplace_back(1);
        } else { outputs.emplace_back(0); }

        planetIDList->emplace_back(cities[i].id);

        for (int j = 0; j < cities[i].distances.size(); ++j) {
            outputs.emplace_back(cities[i].distances[j]);
        }
//        for (int j = 0; j < cities[i].deltaDistances.size(); ++j) {
//            outputs.emplace_back(cities[i].deltaDistances[j]);
//        }
    }

    return InputLayer(&outputs);
}

Matrix<double, Dynamic, Dynamic> GetRandomBiases(vector<int> layerSizes, int numLayers) {
    Matrix<double, Dynamic, Dynamic> biases;

    const int columns = helpers::MaxInArray(&layerSizes);
    const int rows = numLayers;
    biases.resize(rows, columns);

    for (int i = 0; i < columns; i++) {
        for (int j = 0; j < rows; j++) {
            double randomizedBiases = helpers::GetRandomNormalized();

            biases(i, j) = randomizedBiases;
        }
    }

    return biases;
}

Matrix<vector<double>, Dynamic, Dynamic> GetRandomWeights(vector<int> layerSizes, int numLayers, int numInputs) {
    Matrix<vector<double>, Dynamic, Dynamic> weights;

    const int columns = helpers::MaxInArray(&layerSizes);
    const int rows = numLayers;
    weights.resize(numLayers, columns);

    for (int i = 0; i < columns; i++) {
        int layerInputCount = i - 1 < 0 ? numInputs : layerSizes[i - 1];

        for (int j = 0; j < rows; j++) {
            vector<double> randomizedWeights;
            randomizedWeights.resize(layerInputCount);

            for (int k = 0; k < layerInputCount; k++) {
                randomizedWeights[k] = helpers::GetRandomNormalized();
            }

            weights(i, j) = randomizedWeights;
        }
    }

    return weights;
}

int main() {
    auto start = chrono::system_clock::now();

    UpdateUniverseConstants();
    InitializePlanets();

    vector<int> planetIDList;
    InputLayer inputs = SetNetworkInputs(&planetIDList);
    vector<int> layerSizes = {2, 4, 4, 3};
    Matrix<vector<double>, Dynamic, Dynamic> weights = GetRandomWeights(layerSizes, (int)layerSizes.size(), (int)inputs.outputs.size());
    Matrix<double, Dynamic, Dynamic> biases = GetRandomBiases(layerSizes, (int)layerSizes.size());

    NeuralNetworkConfiguration config = NeuralNetworkConfiguration(layerSizes, inputs, weights, biases, planetIDList);
    NeuralNetwork<4> neuralNetwork = NeuralNetwork<4>(&neuralNetwork, &config);

    neuralNetwork.Solve(&neuralNetwork);

    for (int i = 0; i < neuralNetwork.layers[neuralNetwork.size-1].outputs.size(); i++) {
        cout << to_string(neuralNetwork.layers[neuralNetwork.size-1].outputs[i].activation) << endl;
    }

    int newPosition = Neuron::GetHighestNeuronActivationById(&neuralNetwork.layers[1].outputs);

    cout << "New Position is: " + to_string(newPosition) << endl;

    long elapsed_seconds = helpers::GetDuration(start);

    cout << "Executed successfully in " + to_string(elapsed_seconds) + "s!\n";
}
