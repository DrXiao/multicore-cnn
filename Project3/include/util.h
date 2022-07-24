#ifndef __UTIL_H__
#define __UTIL_H__
#define __CPU__ "CPU"
#define __SOBEL__ "sobel"
#define __GAUSSIAN__ "gaussian"
#define IMG_DIR "img/"
#define RESULT_DIR "result/"

void img_dump(int, char *, int, int, const char *);

void cnn(char *, int);

#endif
