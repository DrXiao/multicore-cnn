#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
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

void *convolution_by_CPU(void *convol_para) {
    int thread_idx = ((struct convol_para *)convol_para)->thread_idx;
    int compute_num = ((struct convol_para *)convol_para)->compute_num;
    unsigned char *img = ((struct convol_para *)convol_para)->img;
    int img_len = ((struct convol_para *)convol_para)->img_len;
    unsigned char (*kernel)(unsigned char *, int) = ((struct convol_para *)convol_para)->kernel;
    unsigned char *new_img = ((struct convol_para *)convol_para)->new_img;

    int new_len;
    int offset;
    int upper_bound;
    int start_idx;
    if (kernel == sobel_compute) {
        new_len = img_len - (3 - 1);
        offset = 3 >> 1;
    }
    else {
        new_len = img_len - (5 - 1);
        offset = 5 >> 1;
    }

    upper_bound = min(new_len * new_len, (thread_idx + 1) * compute_num);

    for (start_idx = thread_idx * compute_num; start_idx < upper_bound;
         start_idx++) {
        new_img[start_idx] =
            kernel(img + (start_idx / new_len + offset) * img_len +
                       (start_idx % new_len + offset),
                   img_len);
    }
    return NULL;
}