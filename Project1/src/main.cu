#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>
#include "util.h"

int main(int argc, char **argv) {

    if (argc < 2) {
        printf("Utility:\n\tmain [image.bin]\n");
        exit(0);
    }
    printf("==== Doing Convolution for all files ====\n");
    printf("Thread Block %d, Threads %d\n", THREAD_BLOCK, THREADS);
    for (int img_idx = 1; img_idx < argc; img_idx++)
        cnn(argv[img_idx]);
    printf("==== Ending Convolution for all files ====\n");
    return 0;
}
