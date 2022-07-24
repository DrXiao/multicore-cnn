#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cnn_gpu.h"
#include "util.h"
#define START_IDX blockDim.x *blockIdx.x + threadIdx.x

__device__ unsigned char sobel_gpu_compute(unsigned char *img, int img_len) {
    static char sobel_gpu[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    int result;
    result = 0;
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            result +=
                sobel_gpu[row * 3 + col] * img[(row - 1) * img_len + (col - 1)];
        }
    }
    if (result > 255)
        result = 255;
    else if (result < 0)
        result = 0;
    return (unsigned char)result;
}

__device__ unsigned char gaussian_gpu_compute(unsigned char *img, int img_len) {

    static unsigned char gaussian_gpu[25] = {1,  4, 6,  4,  1,  4, 16, 24, 16,
                                             4,  6, 24, 36, 24, 6, 4,  16, 24,
                                             16, 4, 1,  4,  6,  4, 1};
    int result;
    result = 0;
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 5; col++) {
            result += gaussian_gpu[row * 5 + col] *
                      img[(row - 2) * img_len + (col - 2)];
        }
    }
    result = result >> 8;
    if (result > 255)
        result = 255;
    else if (result < 0)
        result = 0;
    return (unsigned char)result;
}

__global__ void convolution_by_sobel_GPU(unsigned char *image, int img_len,
                                         unsigned char *return_img) {
    // TODO

    int new_len;
    int offset;

    new_len = img_len - 2;
    offset = 1;

    // Doing Convolution
    /*
        for (int new_row = START_IDX; new_row < new_len; new_row += THREADS)
       { for (int new_col = 0; new_col < new_len; new_col++) {
                // res = gaussian_gpu_compute(origin_img_ptr, img_len);
                return_img[new_row * new_len + new_col] = sobel_gpu_compute(
                    image + (new_row + offset) * img_len + (new_col + offset),
                    img_len);
            }
        }
    */

    /*
        ex :    4 Blocks, 2 Threads per block.
                original image : 17 * 17

                new image : 15 * 15 = 225

        total_threads = 4 * 2 = 8
        jump = (15 * 15) / 8 = 28       // (15 * 15) % 8 = 1
        jump += 1      ==> 29

        1st thread
            idx = [0, 29)
        2st thread
            idx = [29, 58)
        .
        .
        .
        8th thread
            idx = [203, 225)            // min((8 - 1) * 29, 15 * 15) =>
       min(232, 225)
    */

    int total_threads = THREADS * THREAD_BLOCK;
    int jump = (new_len * new_len) / total_threads;
    jump += (new_len * new_len) % total_threads == 0 ? 0 : 1;
    int startIdx = START_IDX;
    for (int idx = startIdx * jump;
         idx < min((startIdx + 1) * jump, new_len * new_len); idx++) {
        return_img[idx] =
            sobel_gpu_compute(image + (idx / new_len + offset) * img_len +
                                  (idx % new_len + offset),
                              img_len);
    }
}

__global__ void convolution_by_gaussian_GPU(unsigned char *image, int img_len,
                                            unsigned char *return_img) {
    // TODO

    int new_len;
    int offset;

    new_len = img_len - 4;
    offset = 2;

    // Doing Convolution
    /*
        for (int new_row = START_IDX; new_row < new_len; new_row += THREADS)
       { for (int new_col = 0; new_col < new_len; new_col++) {
                // res = gaussian_gpu_compute(origin_img_ptr, img_len);
                return_img[new_row * new_len + new_col] = gaussian_gpu_compute(
                    image + (new_row + offset) * img_len + (new_col + offset),
                    img_len);
            }
        }
    */

    int total_threads = THREADS * THREAD_BLOCK;
    int jump = (new_len * new_len) / total_threads;
    jump += (new_len * new_len) % total_threads == 0 ? 0 : 1;
    int startIdx = START_IDX;
    for (int idx = startIdx * jump;
         idx < min((startIdx + 1) * jump, new_len * new_len); idx++) {
        return_img[idx] =
            gaussian_gpu_compute(image + (idx / new_len + offset) * img_len +
                                     (idx % new_len + offset),
                                 img_len);
    }
}