//
// Created by Joshua Riefman on 2023-09-04.
//

#include "../include/exceptions.h"

ChrysanthemumExceptions::InvalidConfiguration::InvalidConfiguration(const std::string& message) {
    const int length = (int)message.length();
    char* char_array = new char[length + 1];
    strcpy(char_array, message.c_str());
    this->message = char_array;
}

char* ChrysanthemumExceptions::InvalidConfiguration::what() {
    return message;
}

ChrysanthemumExceptions::PrematureAccess::PrematureAccess(const std::string& message) {
    const int length = (int)message.length();
    char* char_array = new char[length + 1];
    strcpy(char_array, message.c_str());
    this->message = char_array;
}

char* ChrysanthemumExceptions::PrematureAccess::what() {
    return message;
}