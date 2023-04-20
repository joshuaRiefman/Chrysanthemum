cmake_minimum_required(VERSION 3.0.0)
project(Chrysanthemum VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(Chrysanthemum src/Chrysanthemum.cpp src/NeuralNetwork.cpp src/NeuralNetwork.h src/Layer.cpp src/Layer.h src/Neuron.cpp src/Neuron.h src/helpers.cpp src/helpers.h src/Chrysanthemum.h src/City.cpp src/City.h)

find_package(jsoncpp CONFIG REQUIRED)
find_package(eigen3 CONFIG REQUIRED)

target_link_libraries(Chrysanthemum PRIVATE jsoncpp_object jsoncpp_static JsonCpp::JsonCpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)