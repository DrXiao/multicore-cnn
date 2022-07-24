CC := gcc
INCLUDE := include
FLAGS := -g -o
EXECUTABLE := main
INPUT = 6000.bin
LIB = pthread
SRC := src

RESULT := result
MV := mv
RM := rm

THREAD_NUM := 1
INPUT := 6000.bin

all:
	$(CC) $(FLAGS) $(EXECUTABLE) $(SRC)/*.c -I$(INCLUDE) -l$(LIB)

clean:
	$(RM) $(RESULT)/*.bin
	$(RM) $(EXECUTABLE)
