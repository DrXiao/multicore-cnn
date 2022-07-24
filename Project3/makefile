CC := gcc
INCLUDE := include
FLAGS := -g -o
EXECUTABLE := main
INPUT := 6000.bin
OPENMP := openmp
SRC := src

RESULT := result
MV := mv
RM := rm

THREAD_NUM := 1
INPUT := 6000.bin

all:
	$(CC) -f$(OPENMP) $(FLAGS) $(EXECUTABLE) $(SRC)/*.c -I$(INCLUDE)

clean:
	$(RM) $(RESULT)/*
	$(RM) $(EXECUTABLE)
