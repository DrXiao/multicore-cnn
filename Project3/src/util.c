#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include "util.h"
#include "cnn_cpu.h"


void img_dump(int origin_len, char *img, int img_len, int thread_num,
              const char *kernel) {
    FILE *new_file;
    char filename[64];
    sprintf(filename, RESULT_DIR "%s_%d.bin", kernel, origin_len);
    new_file = fopen(filename, "wb");
    for (int i = 0; i < (img_len * img_len); i++) {
        fprintf(new_file, "%c", img[i]);
    }
    fclose(new_file);
}

void cnn(char *filename, int thread_num) {
    FILE *image;
    unsigned char *origin_img, *new_img;
    int origin_len, new_len;
    int img_idx;
    char file[32] = IMG_DIR;
    strcat(file, filename);
    image = fopen(file, "r");
    printf("== %d threads, processing %s ==\n", thread_num, file);
    if (image == NULL) {
        printf("%s\n", file);
        printf("Error! No image!\n");
        return;
    }
    sscanf(filename, "%d.bin", &origin_len);
    origin_img = (unsigned char *)calloc(origin_len * origin_len, sizeof(unsigned char));

    img_idx = 0;
    while (fscanf(image, "%c", origin_img + img_idx) != EOF) {
        img_idx++;
    }

    new_img = convolution_by_CPU(origin_img, origin_len, thread_num, sobel_compute, &new_len);
    //printf("%20s : %ld\n", "Sobel kernel", TIME_DIFF(start, end));
    img_dump(origin_len, new_img, new_len, thread_num, __SOBEL__);
    free(new_img);
    /* Ending Sobel Kernel*/

    /* Starting Gaussian blurred Kernel */
    new_img = convolution_by_CPU(origin_img, origin_len, thread_num, gaussian_compute, &new_len);
    //printf("%20s : %ld\n", "Gaussian kernel", TIME_DIFF(start, end));
    img_dump(origin_len, new_img, new_len, thread_num, __GAUSSIAN__);
    free(new_img);
    /* Ending Gaussian blurred Kernel*/

    free(origin_img);
    fclose(image);
}