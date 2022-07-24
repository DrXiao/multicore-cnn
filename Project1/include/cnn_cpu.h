#ifndef __CNN_CPU_H__
#define __CNN_CPU_H__

unsigned char sobel_compute(unsigned char *, int);

unsigned char gaussian_compute(unsigned char *, int);

unsigned char *convolution_by_CPU(unsigned char *, int, unsigned char (*)(unsigned char *, int), int *);

#endif
