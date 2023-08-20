cmake_minimum_required(VERSION 3.0.0)
project(Chrysanthemum VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)

add_executable(Chrysanthemum src/Chrysanthemum.cpp src/NeuralNetwork.cpp include/NeuralNetwork.h src/Layer.cpp include/Layer.h src/Neuron.cpp include/Neuron.h src/helpers.cpp include/helpers.h include/Chrysanthemum.h src/City.cpp include/City.h)

find_package(jsoncpp CONFIG REQUIRED)
find_package(eigen3 CONFIG REQUIRED)

target_link_libraries(Chrysanthemum PRIVATE jsoncpp_object jsoncpp_static JsonCpp::JsonCpp)