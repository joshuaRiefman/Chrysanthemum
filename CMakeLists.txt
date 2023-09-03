# Setup
cmake_minimum_required(VERSION 3.0.0)
project(Chrysanthemum VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)

enable_testing()

# Application
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
find_package(eigen3 CONFIG REQUIRED)

# GTest and Testing
set(tests ${tests}
        tests/Chrysanthemum_tests.cpp)
find_package(GTest REQUIRED)

add_executable(RunTests tests/tests.h tests/tests.cpp ${tests})
target_link_libraries(RunTests PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(RunTests)