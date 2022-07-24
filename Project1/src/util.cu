#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util.h"
#include "cnn_cpu.h"
#include "cnn_gpu.h"
#define __CPU__ "CPU"
#define __GPU__ "GPU"
#define __SOBEL__ "sobel"
#define __GAUSSIAN__ "gaussian"

void img_dump(int origin_len, unsigned char *img, int img_len, const char *compute_type,
              const char *kernel) {
    FILE *new_file;
    char filename[64];
    sprintf(filename, "%s_%s_%d.bin", compute_type, kernel, origin_len);
    new_file = fopen(filename, "wb");
    for (int i = 0; i < (img_len * img_len); i++) {
        fprintf(new_file, "%c", img[i]);
    }
    fclose(new_file);
}

void cnn(char *argv) {
    int image_len = 0;
    sscanf(argv, "img/%d.bin", &image_len);

    FILE *image = fopen(argv, "rb");
    printf("== %s ==\n", argv);
    if (image == NULL) {
        printf("%s\n", argv);
        printf("Error! No image!\n");
        return;
    }

    unsigned char *image_arr = NULL;
    unsigned char *cuda_copy_img = NULL;
    image_arr = (unsigned char *)calloc(image_len * image_len, sizeof(unsigned char));

    int idx = 0;

    while (fscanf(image, "%c", image_arr + idx) != EOF) {
        idx++;
    }
    cudaMalloc((void **)&cuda_copy_img, image_len * image_len * sizeof(unsigned char));
    cudaMemcpy((void *)cuda_copy_img, (void *)image_arr,
               image_len * image_len * sizeof(unsigned char), cudaMemcpyHostToDevice);

               unsigned char *cpu_new_img = NULL, *gpu_new_img = NULL;
               unsigned char *return_img = NULL;
    int new_len = 0;
    float gpu_elapsed_time = 0;
    cudaEvent_t start, end;

    /* Sobel kernel by CPU */
    printf("%20s - ", "Sobel Kernel CPU");
    cpu_new_img =
        convolution_by_CPU(image_arr, image_len, sobel_compute, &new_len);
    img_dump(image_len, cpu_new_img, new_len, __CPU__, __SOBEL__);
    /* End Sobel kernel by CPU */
    /* Sobel kernel by GPU */
    gpu_new_img = (unsigned char *)malloc(new_len * new_len * sizeof(unsigned char));
    cudaMalloc((void **)&return_img, new_len * new_len * sizeof(unsigned char));
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    convolution_by_sobel_GPU<<<THREAD_BLOCK, THREADS>>>(
        cuda_copy_img, image_len, return_img);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&gpu_elapsed_time, start, end);
    cudaMemcpy((void *)gpu_new_img, (void *)return_img,
               new_len * new_len * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    printf("%20s - ", "Sobel Kernel GPU");
    printf("%d -> %d : %3.20f s\n", image_len, new_len,
           (float)(gpu_elapsed_time) / (float)1000);
    img_dump(image_len, gpu_new_img, new_len, __GPU__, __SOBEL__);
    /* End Sobel kernel by GPU */

    cudaFree(return_img);
    free(cpu_new_img);
    free(gpu_new_img);

    /* Gaussian kernel by CPU */
    printf("%20s - ", "Gaussian Kernel CPU");
    cpu_new_img =
        convolution_by_CPU(image_arr, image_len, gaussian_compute, &new_len);
    img_dump(image_len, cpu_new_img, new_len, __CPU__, __GAUSSIAN__);
    /* End Gaussian kernel by CPU *
    /* Gaussian kernel by GPU */
    gpu_new_img = (unsigned char *)malloc(new_len * new_len * sizeof(unsigned char));
    cudaMalloc((void **)&return_img, new_len * new_len * sizeof(unsigned char));
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    convolution_by_gaussian_GPU<<<THREAD_BLOCK, THREADS>>>(
        cuda_copy_img, image_len, return_img);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&gpu_elapsed_time, start, end);
    cudaMemcpy((void *)gpu_new_img, (void *)return_img,
               new_len * new_len * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    printf("%20s - ", "Gaussian Kernel GPU");
    printf("%d -> %d : %3.20f s\n", image_len, new_len,
           (float)(gpu_elapsed_time) / (float)1000);
    img_dump(image_len, gpu_new_img, new_len, __GPU__, __GAUSSIAN__);
    /* End Gaussian kernel by GPU */
    cudaFree(return_img);
    free(cpu_new_img);
    free(gpu_new_img);

    cudaFree(cuda_copy_img);
    free(image_arr);
    fclose(image);
}
