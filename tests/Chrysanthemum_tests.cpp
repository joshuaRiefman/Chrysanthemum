//
// Created by Joshua Riefman on 2023-09-01.
//

#include "Chrysanthemum_tests.h"
#include "../external/cpputest/include/CppUTest/TestHarness.h"

TEST_GROUP(FirstTestGroup)
        {
        };

TEST(FirstTestGroup, FirstTest)
{
    FAIL("Fail me!");
}