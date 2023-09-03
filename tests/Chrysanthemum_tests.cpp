//
// Created by Joshua Riefman on 2023-09-01.
//

#include "Chrysanthemum_tests.h"

TEST(HelloTest1, BasicAssertions1) {
    // Expect two strings not to be equal.
    EXPECT_STRNE("hello", "world");
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);
}