/* File common.h */
#ifndef COMMON_H
#define COMMON_H

#include <stdint.h> /* uint64_t */
#include <stdio.h>  /* size_t */

#define LOG_ERROR(format, ...) fprintf(stderr, format, __VA_ARGS__)

/* number of input, hidden and output nodes */
#define INODES 784
#define HNODES 200
#define ONODES 10

double Wih[HNODES][INODES];
double Who[ONODES][HNODES];

const char *human_size(uint64_t bytes);
int readcsv(const char *csvfile, int rows, int cols, double *arr);
void printm(double *arr, int rows, int cols);
void printv(double *vec, int length);
double sigmoid(double n);
void mvp(int rows, int cols, double *matrix, double *vector, double *resvec);
void map(double (*func)(double), int length, double *vector, double *result);
void query(double *inputs, double *final_outputs);
void fill_random(double *arr, int rows, int cols);
void vsubtv(double *vector1, double *vector2, double *result, int length);
void transpm(double *arr, int rows, int cols, double *trans);
void vmultv(double *vector1, double *vector2, double *result, int length);
void oneminusv(double *vec, int length);
void vxv(int rows, int cols, double *vector1, double *vector2, double *matrix);
void smultm(int rows, int cols, double *matrix, double scalar);
void mea(int rows, int cols, double *matrix1, double *matrix2);
void train(double *inputs, double *targets);
char *getname(const char *matrix, int rows, int cols);
void tocsv(const char *csvfile, size_t rows, size_t cols,
           double arr[rows][cols]);
void fromcsv(const char *csvfile, size_t rows, size_t cols,
             double arr[rows][cols]);
#endif
