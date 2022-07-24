#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include "util.h"
#include "cnn_cpu.h"
#define TIME_DIFF(start, end)                                                  \
    (1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec)
#define __CPU__ "CPU"
#define __SOBEL__ "sobel"
#define __GAUSSIAN__ "gaussian"
#define IMG_DIR "img/"
#define RESULT_DIR "result/"
#define MAX_THREAD 16

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
    struct timeval start, end;
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

    pthread_t threads[MAX_THREAD];
    struct convol_para convol_para[MAX_THREAD];
    int compute_num;
    /* Starting Sobel Kernel */
    new_len = origin_len - 2;
    new_img = (unsigned char *)malloc(new_len * new_len * sizeof(unsigned char));
    compute_num = (new_len * new_len) / thread_num;
    if ((new_len * new_len) % thread_num)
        compute_num += 1;
    gettimeofday(&start, NULL);
    for (int i = 0; i < thread_num; i++) {
        convol_para[i] = (struct convol_para){.compute_num = compute_num,
                                              .img = origin_img,
                                              .img_len = origin_len,
                                              .kernel = sobel_compute,
                                              .new_img = new_img,
                                              .thread_idx = i};
        if (pthread_create(&threads[i], NULL, convolution_by_CPU,
                           (void *)(&convol_para[i])))
            printf("Error when creating\n");
    }

    for (int i = 0; i < thread_num; i++) {
        if (pthread_join(threads[i], NULL) != 0)
            printf("Error when joining\n");
    }
    gettimeofday(&end, NULL);
    printf("%20s : %ld\n", "Sobel kernel", TIME_DIFF(start, end));
    img_dump(origin_len, new_img, new_len, thread_num, __SOBEL__);
    free(new_img);
    /* Ending Sobel Kernel*/

    /* Starting Gaussian blurred Kernel */
    new_len = origin_len - 4;
    new_img = (unsigned char *)malloc(new_len * new_len * sizeof(unsigned char));
    compute_num = (new_len * new_len) / thread_num;
    if ((new_len * new_len) % thread_num)
        compute_num += 1;
    gettimeofday(&start, NULL);
    for (int i = 0; i < thread_num; i++) {
        convol_para[i] = (struct convol_para){.compute_num = compute_num,
                                              .img = origin_img,
                                              .img_len = origin_len,
                                              .thread_idx = i,
                                              .kernel = gaussian_compute,
                                              .new_img = new_img};
        pthread_create(threads + i, NULL, convolution_by_CPU,
                       (void *)(convol_para + i));
    }
    for (int i = 0; i < thread_num; i++) {
        pthread_join(threads[i], NULL);
    }
    gettimeofday(&end, NULL);
    printf("%20s : %ld\n", "Gaussian kernel", TIME_DIFF(start, end));
    img_dump(origin_len, new_img, new_len, thread_num, __GAUSSIAN__);

    free(new_img);
    /* Ending Gaussian blurred Kernel*/

    free(origin_img);
    fclose(image);
}