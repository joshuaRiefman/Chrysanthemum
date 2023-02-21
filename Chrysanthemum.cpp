#include "/Users/joshuariefman/vcpkg/installed/arm64-osx/include/json/value.h"
#include "/Users/joshuariefman/vcpkg/installed/arm64-osx/include/json/reader.h"
#include "/Users/joshuariefman/vcpkg/installed/arm64-osx/include/json/writer.h"
#include "/Users/joshuariefman/vcpkg/installed/arm64-osx/include/json/json.h"
#include "Neuron.h"
#include "Layer.h"
#include "helpers.h"
#include "NeuralNetwork.h"
#include "Chrysanthemum.h"

using namespace std;


vector<float> &GetArrayFromJSON(int planetID, const Json::Value *data, vector<float> &distances, const string &fieldAccessor) {
    for (int i = 0; i < universeSize; ++i) {
        distances[i] = (*data)["Planets"][planetID][fieldAccessor][i].asFloat();
    }
    return distances;
}

City ParseCityData(int cityID, int citiesCount) {
    ifstream filePath(DataJSONPath);
    Json::Reader reader;
    Json::Value data;

    reader.parse(filePath, data);
    filePath.close();

    int id = cityID;
    double distanceFromOrigin = data["distance_from_origin"].asDouble();

    vector<float> distances;
    distances.resize(citiesCount);
    distances = GetArrayFromJSON(cityID, &data, distances, "distances");

    vector<float> deltaDistances;
    deltaDistances.resize(citiesCount);
    deltaDistances = GetArrayFromJSON(cityID, &data, deltaDistances, "deltaDistances");

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

void InitializePlanets(int citiesCount) {
    cities.resize(citiesCount);
    for (int i = 0; i < citiesCount; i++)
    {
        cities[i] = ParseCityData(i, citiesCount);
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

int main() {
    auto start = chrono::system_clock::now();

    UpdateUniverseConstants();
    InitializePlanets(universeSize);

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
