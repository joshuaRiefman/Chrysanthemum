#include "../include/Chrysanthemum.h"

void print() {
    std::cout << "Here!" << "\n";
}

std::vector<float> &GetArrayFromJSON(int planetID, const Json::Value *data, std::vector<float> &distances, const std::string &fieldAccessor) {
    for (int i = 0; i < universeSize; ++i) {
        distances[i] = (*data)["Planets"][planetID][fieldAccessor][i].asFloat();
    }
    return distances;
}

City ParseCityData(int cityID, int citiesCount) {
    std::ifstream filePath(DataJSONPath);
    Json::Reader reader;
    Json::Value data;

    reader.parse(filePath, data);
    filePath.close();

    int id = cityID;
    double distanceFromOrigin = data["distance_from_origin"].asDouble();

    std::vector<float> distances;
    distances.resize(citiesCount);
    distances = GetArrayFromJSON(cityID, &data, distances, "distances");

    std::vector<float> deltaDistances;
    deltaDistances.resize(citiesCount);
    deltaDistances = GetArrayFromJSON(cityID, &data, deltaDistances, "deltaDistances");

    City newCity(id, distances, distanceFromOrigin, deltaDistances, false);
    return newCity;
}

void UpdateUniverseConstants() {
    Json::StreamWriterBuilder builder;
    builder["commentStyle"] = "None";
    builder["indentation"] = "\t";

    Json::Value value;
    value["universeSize"] = (int)universeSize;

    std::ofstream outputFileStream(ConstantsJSONPath);
    builder.newStreamWriter()->write(value, &outputFileStream);

    outputFileStream.close();
}

void SetOrigin() {
    std::ifstream filePath(DataJSONPath);
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

static std::shared_ptr<InputLayer> SetNetworkInputs() {
    std::vector<double> outputs;
    for (int i = 0; i < 3; ++i) {
//
//        if (cities[i].visited) {
//            outputs.emplace_back(1);
//        } else { outputs.emplace_back(0); }
        for (float & distance : cities[i].distances) {
            outputs.emplace_back(distance);
        }
//        for (int j = 0; j < cities[i].deltaDistances.size(); ++j) {
//            outputs.emplace_back(cities[i].deltaDistances[j]);
//        }
    }
    return std::make_shared<InputLayer>(outputs);
}

void InitializeWorld() {
    UpdateUniverseConstants();
    InitializePlanets(universeSize);
}

int main() {
    InitializeWorld();

//    std::shared_ptr<InputLayer> inputLayer = SetNetworkInputs();
    std::vector<double> outputs = {1,1,1,1};
    std::shared_ptr<InputLayer> inputLayer = std::make_shared<InputLayer>(outputs);
    std::vector<int> layerSizes = {2, 2, 2};
//    using new_weight_t = std::vector<std::shared_ptr<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>>;
    weight_t in_weights = GetRandomWeights(layerSizes, (int)layerSizes.size(), (int)inputLayer->numInputs);
    bias_t in_biases = GetRandomBiases(layerSizes, (int)layerSizes.size());
    NeuralNetwork neuralNetwork = NeuralNetwork(inputLayer->numInputs, layerSizes, in_weights, in_biases);

    neuralNetwork.SetInputs(inputLayer);
    neuralNetwork.Solve();

    for (const auto & value : neuralNetwork.GetOutputVector()) {
        std::cout << std::to_string(value) << "\n";
    }

    int newPosition = neuralNetwork.GetHighestNeuronActivationById();

    std::cout << "New Position is: " + std::to_string(newPosition) << "\n";
    std::cout << "Executed successfully! " <<  std::endl;
}
