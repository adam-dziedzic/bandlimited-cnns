//
// Created by Adam Dziedzic on 7/11/18.
//

#include <iostream>
#include "Utils.h"

void Utils::printTable(double *table, int len) {
    for (int i = 0; i < len; ++i) {
        std::cout << table[i] << " ";
    }
}