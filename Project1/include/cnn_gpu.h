#ifndef __CNN_GPU_H__
#define __CNN_GPU_H__

__device__ unsigned char sobel_gpu_compute(unsigned char *, int);

__device__ unsigned char gaussian_gpu_compute(unsigned char *, int);

__global__ void convolution_by_sobel_GPU(unsigned char *, int, unsigned char *);

__global__ void convolution_by_gaussian_GPU(unsigned char *, int, unsigned char *);

#endif
