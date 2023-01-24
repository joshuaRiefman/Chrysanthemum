cmake_minimum_required(VERSION 3.0.0)
project(Chrysanthemum VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(Chrysanthemum Chrysanthemum.cpp NeuralNetwork.h NeuralNetwork.cpp)

set(CMAKE_TOOLCHAIN_FILE "-DCMAKE_TOOLCHAIN_FILE=/Users/joshuariefman/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(eigen_DIR "/Users/joshuariefman/Nostradamus/Chrysanthemum/Packages/eigen-3.4.0")
include(/Users/joshuariefman/vcpkg/scripts/buildsystems/vcpkg.cmake)
find_package(jsoncpp CONFIG REQUIRED)
find_package(eigen3 CONFIG REQUIRED)
target_link_libraries(Chrysanthemum PRIVATE jsoncpp_object jsoncpp_static JsonCpp::JsonCpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)