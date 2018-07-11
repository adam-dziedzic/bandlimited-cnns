//
// Created by Adam Dziedzic on 7/11/18.
//

#ifndef CFFT_FFT_H
#define CFFT_FFT_H


class FFT {

public:
    /**
     * Discrete Fourier Transform. This computes an in-place (so we do not have to bother with who should realse the
     * memory problem) DFT.
     *
     * @param dir - direction of the transformation: dir = 1 gives forward transofrm, dir = -1 gives reverse transform
     * @param m - the size of the x and y arrays
     * @param x - an array representing the real part of the signal
     * @param y - an array representing the imaginary part of the signal
     *
     * @return 0 if everything went okay, 1 otherwise.
     */
    int dft(int dir, int m, double *x, double *y);
};


#endif //CFFT_FFT_H
