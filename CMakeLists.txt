cmake_minimum_required(VERSION 3.0.0)
project(Chrysanthemum VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)

set(sources ${sources}
        src/Chrysanthemum.cpp
        src/NeuralNetwork.cpp
        src/Layer.cpp
        src/helpers.cpp
        )
set(includes ${includes}
        include/Chrysanthemum.h
        include/helpers.h
        include/Layer.h
        include/NeuralNetwork.h
        )

add_executable(Chrysanthemum src/main.cpp ${sources} ${includes})
add_executable(Chrysanthemum_Tests tests/Chrysanthemum_tests.cpp tests/Chrysanthemum_tests.h ${sources} ${includes})
add_executable(NeuralNetwork_Tests tests/NeuralNetwork_tests.cpp tests/NeuralNetwork_tests.h ${sources} ${includes})

find_package(jsoncpp CONFIG REQUIRED)
find_package(eigen3 CONFIG REQUIRED)

target_link_libraries(Chrysanthemum PRIVATE jsoncpp_object jsoncpp_static JsonCpp::JsonCpp)
target_link_libraries(Chrysanthemum_Tests PRIVATE jsoncpp_object jsoncpp_static JsonCpp::JsonCpp)
target_link_libraries(NeuralNetwork_Tests PRIVATE jsoncpp_object jsoncpp_static JsonCpp::JsonCpp)

