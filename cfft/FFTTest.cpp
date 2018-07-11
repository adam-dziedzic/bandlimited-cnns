//
// Created by Adam Dziedzic on 7/11/18.
//
#include <iostream>
#include "FFT.h"
#include <gtest/gtest.h>
#include "Utils.h"

void compareTables(double *tableResult, double *tableExpected, int len) {
    for (int i = 0; i < len; ++i) {
        std::cout << "i: " << i << " actual: " << tableResult[i] << " expected: " << tableExpected[i] << std::endl;
        ASSERT_NEAR(tableResult[i], tableExpected[i], 0.00000001);
    }
}

TEST(FFTTest, SimpleTest) {
    int size = 4;
    double x1[size] = {0.0, 1.0, 2.0, 3.0};
    double y1[size] = {0.0, 0.0, 0.0, 0.0};

    double xExpected[size] = {6.0, -2.0, -2.0, -2.0};
    double yExpected[size] = {0.0, 2.0, 0.0, -2.0};

    std::cout << "x1: ";
    Utils::printTable(x1, size);
    std::cout << std::endl;
    std::cout << "y1: ";
    Utils::printTable(y1, size);

    FFT fft = FFT();
    fft.dft(1, size, x1, y1);

    std::cout << "the result of fft: " << std::endl;
    std::cout << "x1: ";
    Utils::printTable(x1, size);
    compareTables(x1, xExpected, size);
    std::cout << std::endl;
    std::cout << "y1: ";
    Utils::printTable(y1, size);
    std::cout << std::endl;
    compareTables(y1, yExpected, size);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}