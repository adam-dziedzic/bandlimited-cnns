//
// Created by Adam Dziedzic on 7/11/18.
//

#include "FFT.h"
#include <cmath>

int FFT::dft(int dir, int m, double *x1, double *y1) {
    long i, k; // the indices for the for loops
    double arg; // the argument to the e^{arg} or for cos(arg) + j sin(arg) representation of complex numbers
    double cosarg, sinarg; // the arguments for the trigonometric functions (for the complex number representation)
    double *x2 = nullptr, *y2 = nullptr; // the temporary storage for the results

    x2 = new double[m * sizeof(double)];
    y2 = new double[m * sizeof(double)];

    // compute the output for the whole signal
    for (i = 0; i < m; ++i) {
        x2[i] = 0;
        y2[i] = 0;
        arg = -dir * 2.0 * 3.141592654 * (double) i / (double) m;
        // compute a single coefficient (for either time or frequency domain)
        for (k = 0; k < m; ++k) {
            cosarg = cos(k * arg);
            sinarg = sin(k * arg);
            // we express e^{arg} = cos(arg) * j*sin(arg)
            // (x1[k] + j*y1[k])(cos(arg) * j*sin(arg)) = x1[k]*cos(arg) + j*j*y1[k]*sin(arg) =
            // x1[k]*cos(arg) - y1[k]*sin(arg)
            x2[i] += (x1[k] * cosarg - y1[k] * sinarg);
            // j(y1[k]*cos(arg) + x1[k]*sin(arg))
            y2[i] += y1[k] * cosarg + x1[k] * sinarg;
        }
    }

    /* Copy the data back. */
    if (dir == -1) {
        for (i = 0; i < m; ++i) {
            x1[i] = x2[i] / (double) m;
            y1[i] = y2[i] / (double) m;
        }
    } else {
        for (i = 0; i < m; ++i) {
            x1[i] = x2[i];
            y1[i] = y2[i];
        }
    }

    delete[] x2;
    delete[] y2;
    return 0;
}

