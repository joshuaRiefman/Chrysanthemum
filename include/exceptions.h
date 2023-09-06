//
// Created by Joshua Riefman on 2023-09-04.
//

#ifndef CHRYSANTHEMUM_EXCEPTIONS_H
#define CHRYSANTHEMUM_EXCEPTIONS_H

#include <exception>
#include <string>

namespace ChrysanthemumExceptions {
    class InvalidConfiguration : public std::exception {
    private:
        char* message;
    public:
        explicit InvalidConfiguration(const std::string& message);

        char* what();
    };

    class PrematureAccess : public std::exception {
    private:
        char* message;
    public:
        explicit PrematureAccess(const std::string& message);

        char* what();
    };
}

#endif//CHRYSANTHEMUM_EXCEPTIONS_H
