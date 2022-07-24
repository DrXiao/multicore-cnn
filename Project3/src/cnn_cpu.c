#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "util.h"
#include "cnn_cpu.h"

unsigned char sobel_compute(unsigned char *img, int img_len) {
    static char sobel[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    int result;
    result = 0;
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            result +=
                sobel[row * 3 + col] * img[(row - 1) * img_len + (col - 1)];
        }
    }
    if (result > 255)
        result = 255;
    else if (result < 0)
        result = 0;
    return (unsigned char)result;
}

unsigned char gaussian_compute(unsigned char *img, int img_len) {

    static char gaussian_blur[25] = {1,  4, 6,  4,  1,  4, 16, 24, 16,
                                     4,  6, 24, 36, 24, 6, 4,  16, 24,
                                     16, 4, 1,  4,  6,  4, 1};
    int result;
    result = 0;
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 5; col++) {
            result += gaussian_blur[row * 5 + col] *
                      img[(row - 2) * img_len + (col - 2)];
        }
    }
    result = (result >> 8);
    if (result > 255)
        result = 255;
    else if (result < 0)
        result = 0;
    return (unsigned char)result;
}

int min(int a, int b) {
    return a <= b ? a : b;
}

unsigned char *convolution_by_CPU(unsigned char *img, int img_len,
                                  int thread_num,
                                  unsigned char (*kernel)(unsigned char *, int),
                                  int *return_len) {

    unsigned char *new_img;
    int offset;
    double time;
    int new_row;
    int new_col;
    if (kernel == sobel_compute) {
        *return_len = img_len - (3 - 1);
        offset = 3 >> 1;
    }
    else {
        *return_len = img_len - (5 - 1);
        offset = 5 >> 1;
    }
    new_img = (unsigned char *)malloc(*return_len * *return_len *
                                      sizeof(unsigned char));
    time = omp_get_wtime();

#pragma omp parallel shared(img, new_img, offset, return_len) private(new_row, new_col) num_threads(thread_num)
    {
        #pragma omp for
        for (new_row = 0; new_row < *return_len; new_row++) {
            for (new_col = 0; new_col < *return_len; new_col++) {
                new_img[new_row * *return_len + new_col] = kernel(
                    img + (new_row + offset) * img_len + (new_col + offset),
                    img_len);
            }
        }
    }
    printf("\n");
    printf("%-20s : %.16g s\n",
           kernel == sobel_compute ? __SOBEL__ : __GAUSSIAN__,
           omp_get_wtime() - time);
    return new_img;
}