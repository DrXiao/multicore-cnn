#ifndef __CNN_GPU_H__
#define __CNN_GPU_H__
#include <pthread.h>

struct convol_para {
    int thread_idx;
    int compute_num;
    unsigned char *img;
    int img_len;
    unsigned char (*kernel)(unsigned char *, int);
    unsigned char *new_img;
};

unsigned char sobel_compute(unsigned char *, int);

unsigned char gaussian_compute(unsigned char *, int);

void *convolution_by_CPU(void *);

// char *convolution_by_CPU(char *, int, char (*)(char *, int), int *);

#endif