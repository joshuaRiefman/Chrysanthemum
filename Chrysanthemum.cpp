#include <string>
#include <iostream>
#include <fstream>
#include <array>
#include <utility>
#include <vector>
#include <random>
#include <chrono>
#include <ctime>
#include "/Users/joshuariefman/vcpkg/installed/arm64-osx/include/json/value.h"
#include "/Users/joshuariefman/vcpkg/installed/arm64-osx/include/json/reader.h"
#include "/Users/joshuariefman/vcpkg/installed/arm64-osx/include/json/writer.h"
#include "/Users/joshuariefman/vcpkg/installed/arm64-osx/include/json/json.h"
#include "/Users/joshuariefman/Nostradamus/Chrysanthemum/Packages/eigen-3.4.0/Eigen/Eigen"
#include "NeuralNetwork.h"

using namespace std;
using namespace Eigen;

const string DataJSONPath = "/Users/joshuariefman/Nostradamus/Chrysanthemum/data.json";
const string ConstantsJSONPath = "/Users/joshuariefman/Nostradamus/Chrysanthemum/constants.json";
const string PathJSONPath = "/Users/joshuariefman/Nostradamus/Chrysanthemum/path.json";
const unsigned long universeSize = 3;

static int Sum(vector<int> *layerSizes) {
    int result = 0;

    for (int i = 0; i < layerSizes->size(); i++) {
        result += (*layerSizes)[i];
    }

    return result;
}

static int MaxInArray(vector<int> *pVector) {
    int max = 0;

    for (int i = 0; i < pVector->size(); i++) {
        if ((*pVector)[i] > max) { max = (*pVector)[i]; }
    }

    return max;
}

double ReLU(double value) {
    return value > 0 ? value : 0;
}

array<float, universeSize> &GetArrayFromJSON(int planetID, const Json::Value *data, array<float, universeSize> &distances, const string &fieldAccessor) {
    for (int i = 0; i < universeSize; ++i) {
        distances[i] = (*data)["Planets"][planetID][fieldAccessor][i].asFloat();
    }
    return distances;
}

double GetRandomNormalized() {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distribution(-1000, 1000);
    return (float)distribution(gen)/1000;
}

class Planet {
    public:
        int id;
        double distanceFromOrigin;
        array<float, universeSize> distances{};
        array<float, universeSize> deltaDistances{};
        bool visited;

        Planet() = default;

        Planet(int ID, array<float, universeSize> DISTANCES, double DISTANCEFROMORIGIN, array<float, universeSize> DELTADISTANCES, bool VISITED) {
            id = ID;
            distances = DISTANCES;
            distanceFromOrigin = DISTANCEFROMORIGIN;
            deltaDistances = DELTADISTANCES;
            visited = VISITED;
        }
};

Planet planets[universeSize];
int originPlanetID;

Planet ParsePlanetData(int planetID) {
    ifstream filePath(DataJSONPath);
    Json::Reader reader;
    Json::Value data;

    reader.parse(filePath, data);
    filePath.close();

    int id = planetID;
    double distanceFromOrigin = data["distance_from_origin"].asDouble();
    array<float, universeSize> distances = GetArrayFromJSON(planetID, &data, distances, "distances");
    array<float, universeSize> deltaDistances = GetArrayFromJSON(planetID, &data, deltaDistances,  "deltaDistances");

    Planet newPlanet(id, distances, distanceFromOrigin, deltaDistances, false);
    return newPlanet;
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

    originPlanetID = data["starting_position"].asInt();
    planets[originPlanetID].visited = true;
 }

void InitializePlanets() {
    for (int i = 0; i < universeSize; i++)
    {
        planets[i] = ParsePlanetData(i);
    }

    SetOrigin();
}

class Neuron {
public:
    Neuron(double activation, const vector<double> &weights, double bias)
    : activation(activation), weights(weights), bias(bias) {}

    Neuron() : activation(0), weights({1, 1, 1}), bias(1.5) {};

    double bias{};//turn into pointer
    double activation = 0;
    vector<double> weights{}; //turn into pointer to matrix
};

class Layer {
public:
    vector<Neuron> outputs;
    vector<double> inputs;

    Layer(int numOutputs, int numInputs) {
        for (int i = 0; i < numOutputs; ++i) {
            outputs.emplace_back();
        }

        for (int i = 0; i < numInputs; ++i) {
            inputs.emplace_back();
        }
    }

    Layer() = default;

    static Layer CreateLayer(int numOutputs, int numInputs) {
        return {numOutputs, numInputs};
    }
};

struct InputLayer : Layer {

    vector<double> d_inputs;

    explicit InputLayer(vector<double> *INPUTS) {
        d_inputs = *INPUTS;

        outputs.resize(d_inputs.size());

        for (int i = 0; i < outputs.size(); ++i) {
            outputs[i].activation = d_inputs[i];
        }
    }

    InputLayer() = default;

};

static InputLayer SetNetworkInputs(vector<int> *planetIDList) {
    vector<double> outputs;
    for (int i = 0; i < universeSize; ++i) {

        if (planets[i].visited) {
            outputs.emplace_back(1);
        } else { outputs.emplace_back(0); }

        planetIDList->emplace_back(planets[i].id);

        for (int j = 0; j < planets[i].distances.size(); ++j) {
            outputs.emplace_back(planets[i].distances[j]);
        }
//        for (int j = 0; j < planets[i].deltaDistances.size(); ++j) {
//            outputs.emplace_back(planets[i].deltaDistances[j]);
//        }
    }

    return InputLayer(&outputs);
}

struct NeuralNetworkConfiguration {
    vector<int> layerSizes{};
    vector<int> planetIDList{};
    InputLayer inputValues{};

    NeuralNetworkConfiguration(const vector<int> &layerSizes, InputLayer inputs,
                               const Matrix<vector<double>, Dynamic, Dynamic> &weights,
                               const Matrix<double, Dynamic, Dynamic> &biases, const vector<int> &planetIDList)
                               : layerSizes(layerSizes), inputValues(std::move(inputs)), weights(weights), biases(biases), planetIDList(planetIDList) {}

    Matrix<vector<double>, Dynamic, Dynamic> weights{};
    Matrix<double, Dynamic, Dynamic> biases{};
};

template<int networkSize>
class NeuralNetwork {
public:
    NeuralNetwork() : layers(), size(networkSize) {}

    Matrix<vector<double>, Dynamic, Dynamic> weights{};
    Matrix<double, Dynamic, Dynamic> biases{};
    Layer layers[networkSize];
    InputLayer inputLayer;
    int size{};

    NeuralNetwork(NeuralNetwork<networkSize> *network, NeuralNetworkConfiguration *config) {
        int maxNeuronCountPerLayer = MaxInArray(&config->layerSizes);
        network->size = networkSize;
        network->weights.resize(network->size, maxNeuronCountPerLayer);
        network->biases.resize(network->size, maxNeuronCountPerLayer);

        network->weights = config->weights;
        network->biases = config->biases;

        network->inputLayer = config->inputValues;

        for (int i = 0; i < networkSize; i++) {
            int numOutputs = config->layerSizes[i];
            int numInputs;
            if (i-1 < 0) { numInputs = (int)config->inputValues.inputs.size(); } else { numInputs = config->layerSizes[i-1]; }

            network->layers[i] = Layer::CreateLayer(numOutputs, numInputs);
        }

        for (int i = 0; i < network->size; i++) {
            for (int j = 0; j < maxNeuronCountPerLayer; j++) {
                network->layers[i].outputs[j].weights = network->weights(i, j);
                network->layers[i].outputs[j].bias = network->biases(i, j);
            }
        }
    }

    void Solve(NeuralNetwork<networkSize> *network) {
        for (int i = 0; i < network->layers[0].inputs.size(); ++i) {
            network->layers[0].inputs[i] = network->inputLayer.outputs[i].activation;
        }

        for (int i = 0; i < networkSize; i++) {
            Layer *layer = &network->layers[i];
            if (i != 0) {
                for (int j = 0; j < layer->inputs.size(); ++j) {
                    layer->inputs[j] = network->layers[i-1].outputs[j].activation;
                }
            }

            for (int j = 0; j < layer->outputs.size(); j++) {
                for (int k = 0; k < layer->inputs.size(); k++) {
                    layer->outputs[j].activation += layer->inputs[k] * layer->outputs[j].weights[k];
                }
                layer->outputs[j].activation += layer->outputs[j].bias;
                layer->outputs[j].activation = ReLU(layer->outputs[j].activation);
            }
        }
    }
};

Matrix<double, Dynamic, Dynamic> GetRandomBiases(vector<int> layerSizes, int numLayers) {
    Matrix<double, Dynamic, Dynamic> biases;

    const int columns = MaxInArray(&layerSizes);
    const int rows = numLayers;
    biases.resize(rows, columns);

    for (int i = 0; i < columns; i++) {
        for (int j = 0; j < rows; j++) {
            double randomizedBiases = GetRandomNormalized();

            biases(i, j) = randomizedBiases;
        }
    }

    return biases;
}

Matrix<vector<double>, Dynamic, Dynamic> GetRandomWeights(vector<int> layerSizes, int numLayers, int numInputs) {
    Matrix<vector<double>, Dynamic, Dynamic> weights;

    const int columns = MaxInArray(&layerSizes);
    const int rows = numLayers;
    weights.resize(numLayers, columns);

    for (int i = 0; i < columns; i++) {
        int layerInputCount = i - 1 < 0 ? numInputs : layerSizes[i - 1];

        for (int j = 0; j < rows; j++) {
            vector<double> randomizedWeights;
            randomizedWeights.resize(layerInputCount);

            for (int k = 0; k < layerInputCount; k++) {
                randomizedWeights[k] = GetRandomNormalized();
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

    auto end = chrono::system_clock::now();
    chrono::duration<double> elapsed_seconds = end-start;
    time_t end_time = std::chrono::system_clock::to_time_t(end);

    cout << "Executed successfully in " + to_string(elapsed_seconds.count()) + "s!\n";
}
