CC := nvcc
INCLUDE := include
FLAGS := -g -o
EXECUTABLE := main
INPUT := *.bin

SRC := src

CPU_IMG := CPU_IMG
GPU_IMG := GPU_IMG

RM := rm
MV := mv

all:
	$(CC) $(FLAGS) $(EXECUTABLE) $(SRC)/*.cu -I$(INCLUDE)
	mkdir $(CPU_IMG)
	mkdir $(GPU_IMG)

run:
	./$(EXECUTABLE) img/$(INPUT)
	$(MV) CPU_*.bin $(CPU_IMG)
	$(MV) GPU_*.bin $(GPU_IMG)

clean:
	$(RM) -rf $(CPU_IMG) $(GPU_IMG)
	$(RM) $(EXECUTABLE)

bitcheck:
	cmp -l $(CPU_IMG)/CPU_sobel_$(INPUT) $(GPU_IMG)/GPU_sobel_$(INPUT) 
	cmp -l $(CPU_IMG)/CPU_gaussian_$(INPUT) $(GPU_IMG)/GPU_gaussian_$(INPUT) 

check:
	python bitmap.py $(CPU_IMG)/CPU_sobel_$(INPUT) 2 $(CPU_IMG)
	python bitmap.py $(GPU_IMG)/GPU_sobel_$(INPUT) 2 $(GPU_IMG)
	python bitmap.py $(CPU_IMG)/CPU_gaussian_$(INPUT) 4 $(CPU_IMG)
	python bitmap.py $(GPU_IMG)/GPU_gaussian_$(INPUT) 4 $(GPU_IMG)

move:
	$(MV) CPU_*.bin $(CPU_IMG)
	$(MV) GPU_*.bin $(GPU_IMG)
