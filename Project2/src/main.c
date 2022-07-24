#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "cnn_cpu.h"

int main(int argc, char **argv) {

    if (argc < 3) {
        printf("Utility\n\tmain thread_num img.bin\n");
        exit(0);
    }

    cnn(argv[2], atoi(argv[1]));

    return 0;
}