#include <iostream>
#include "FFT.h"
#include "Utils.h"

int main() {
    std::cout << "Hello, World!" << std::endl;
    int size = 4;
    double x1[size] = {0.0, 1.0, 2.0, 3.0};
    double y1[size] = {0.0, 0.0, 0.0, 0.0};
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
    std::cout << std::endl;
    std::cout << "y1: ";
    Utils::printTable(y1, size);
}