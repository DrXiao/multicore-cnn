#!/bin/bash

for thread_num in 1 2 3 4 5 6 7 8; do
    for binary in 320 1280 2880 6000 11400; do
        ./main $thread_num $binary.bin >> record/$thread_num.txt
    done
done
