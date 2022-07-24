#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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

    static unsigned char gaussian_blur[25] = {1,  4, 6,  4,  1,  4, 16, 24, 16,
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
    // printf("result : %d\n", result);
    result = result >> 8;
    if (result > 255)
        result = 255;
    else if (result < 0)
        result = 0;
    return (unsigned char)result;
}

unsigned char *convolution_by_CPU(unsigned char *image, int img_len,
                                  unsigned char (*kernel)(unsigned char *, int),
                                  int *return_len) {
    unsigned char *new_img;
    int new_len;
    int offset;
    clock_t start, end;
    start = clock();
    if (kernel == sobel_compute) {
        new_len = img_len - (3 - 1);
        offset = 3 >> 1;
    }
    else {
        new_len = img_len - (5 - 1);
        offset = 5 >> 1;
    }
    new_img =
        (unsigned char *)malloc(new_len * new_len * sizeof(unsigned char));

    // Doing Convolution

    for (int new_row = 0; new_row < new_len; new_row++) {
        for (int new_col = 0; new_col < new_len; new_col++) {
            new_img[new_row * new_len + new_col] = kernel(
                image + (new_row + offset) * img_len + (new_col + offset),
                img_len);
        }
    }
    *return_len = new_len;
    end = clock();
    printf("%d -> %d : %3.20f s\n", img_len, new_len,
           (float)(end - start) / (float)CLOCKS_PER_SEC);
    return new_img;
}
