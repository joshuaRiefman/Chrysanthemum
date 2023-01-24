# Chrysanthemum

## What is Chrysanthemum?

At its core, Chrysanthemum is a neural network that aims to approximate a solution to the travelling salesman problem. Built in C++, Chrysanthemum's end goal is to be able to take in a list of cities and the distances between each, and to find a path to each city and back to the original that minimizes distance travelled. 

## How to use Chrysanthemum

Chrysanthemum is currently in-progress. At the moment, Chrysanthemum's neural network is fully functional, but lacks a learning algorithm which is the next step in its development. To use Chrysanthemum, clone this repository and run the executable. Ulysses (https://github.com/joshuaRiefman/Ulysses) was built for the purpose of providing city data and visualizing the path created by Chrysanthemum, but you can use another program to provide and handle this data. 

## What's next?

The nature of the travelling salesman problem means that the network will have to execute for every city, and that any change in parameters which change the path can dramatically change the entire path that the network takes. This means that for my knowledge of machine learning, it isn't possible to create a simple, differentiable cost function in which to apply backpropagation. An learning technique alternative has been found, but will take a bit of time to implement. In the mean time, Chrysanthemum's dependencies must be formalized and simplified. Furthermore, Chrysanthemum is all contained within one .cpp file, but should be seperated into numerous .cpp and .hpp files. 
