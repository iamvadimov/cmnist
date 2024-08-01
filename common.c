/* File common.c */
#include "common.h"
#include <inttypes.h> // PRIu64
#include <math.h>
#include <stdint.h> // uint64_t
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const double EULER_NUMBER = 2.71828;
const float LEARNING_RATE = 0.1;

const char *human_size(uint64_t bytes) {
  char *suffix[] = {"B", "KB", "MB", "GB", "TB"};
  char length = sizeof(suffix) / sizeof(suffix[0]);

  int i = 0;
  double dbytes = bytes;

  if (bytes > 1024) {
    for (i = 0; (bytes / 1024) > 0 && i < length - 1; i++, bytes /= 1024) {
      dbytes = bytes / 1024.0;
    }
  }

  static char output[200];
  sprintf(output, "%.02lf %s", dbytes, suffix[i]);
  return output;
}

int readcsv(const char *csvfile, int rows, int cols, double *arr) {
  const size_t MAX_LEN = 2 + 28 * 28 * 4;
  FILE *fd = fopen(csvfile, "r");
  if (fd == NULL) {
    LOG_ERROR("Error reading file %s\n.", csvfile);
    return 1;
  }

  double(*M)[rows][cols] = (void *)arr;
  char buf[MAX_LEN] = {0};
  for (int i = 0; i < rows; i++) {
    char *result = fgets(buf, MAX_LEN, fd);
    if (result != NULL) {
      int j = 0;
      char *c = strtok(result, ",");
      while (c != NULL) {
        (*M)[i][j] = (j == 0) ? atoi(c) : atoi(c) / 255.0 * 0.99 + 0.01;
        j++;
        c = strtok(NULL, ",");
      }
    } else {
      break;
    }
  }
  fclose(fd);
  return 0;
}

void printm(double *arr, int rows, int cols) {
  /* Accessing the array values as if it was a 2D array */
  for (int i = 0; i < rows; i++) {
    int row = i * cols;
    for (int j = 0; j < cols; j++) {
      printf("%g\t", arr[row + j]);
    }
    putchar('\n');
  }
  putchar('\n');
}

void printv(double *vec, int length) {
  for (int i = 0; i < length; i++) {
    printf("%.3f\t", vec[i]);
  }
  putchar('\n');
}

double sigmoid(double n) { return (1 / (1 + pow(EULER_NUMBER, -n))); }

void mvp(int rows, int cols, double *matrix, double *vector, double *resvec) {
  /* Matrix-vector product */
  for (int i = 0; i < cols; i++) {
    for (int j = 0; j < rows; j++) {
      resvec[j] += *((matrix + j * cols) + i) * vector[i];
    }
  }
}

void map(double (*func)(double), int length, double *vector, double *result) {
  for (int i = 0; i < length; i++) {
    result[i] = (*func)(vector[i]);
  }
}

/* query the neural network */
void query(double *inputs, double *final_outputs) {
  /* calculate signals into hidden layer */
  double hidden_inputs[HNODES] = {0};
  mvp(HNODES, INODES, (double *)Wih, inputs, hidden_inputs);

  /* calculate the signals emerging from hidden layer */
  double hidden_outputs[HNODES] = {0};
  map(sigmoid, HNODES, hidden_inputs, hidden_outputs);

  /* calculate signals into final output layer */
  double final_inputs[ONODES] = {0};
  mvp(ONODES, HNODES, (double *)Who, hidden_outputs, final_inputs);

  /* calculate the signals emerging from final output layer */
  map(sigmoid, ONODES, final_inputs, final_outputs);
}

void fill_random(double *arr, int rows, int cols) {
  srand(time(NULL));
  for (int i = 0; i < rows; i++) {
    int row = i * cols;
    for (int j = 0; j < cols; j++) {
      *((arr + row) + j) = 1.0 * rand() / RAND_MAX - 0.5; /* -0.5 .. +0.5 */
    }
  }
}

void vsubtv(double *vector1, double *vector2, double *result, int length) {
  for (int i = 0; i < length; i++) {
    result[i] = vector1[i] - vector2[i];
  }
}

void transpm(double *arr, int rows, int cols, double *trans) {
  for (int i = 0; i < cols; i++) {
    int x = i * rows;
    for (int j = 0; j < rows; j++) {
      *((trans + x) + j) = *((arr + j * cols) + i);
    }
  }
}

void vmultv(double *vector1, double *vector2, double *result, int length) {
  for (int i = 0; i < length; i++) {
    result[i] = vector1[i] * vector2[i];
  }
}

void oneminusv(double *vec, int length) {
  for (int i = 0; i < length; i++) {
    vec[i] = 1 - vec[i];
  }
}

void vxv(int rows, int cols, double *vector1, double *vector2, double *matrix) {
  /*
  | 2 |               |  8	10  12 |
  | 3 | x | 4 5	6 | = | 12	15	18 |
  */
  for (int i = 0; i < rows; i++) {
    int row = i * cols;
    for (int j = 0; j < cols; j++) {
      *((matrix + row) + j) = vector1[i] * vector2[j];
    }
  }
}

void smultm(int rows, int cols, double *matrix, double scalar) {
  /* multiply each element in matrix by the given scalar */
  for (int i = 0; i < rows; i++) {
    int row = i * cols;
    for (int j = 0; j < cols; j++) {
      *((matrix + row) + j) *= scalar;
    }
  }
}

void mea(int rows, int cols, double *matrix1, double *matrix2) {
  /* matrix enhanced assignment: matrix1 += matrix2 */
  for (int i = 0; i < rows; ++i) {
    int row = i * cols;
    for (int j = 0; j < cols; ++j) {
      *((matrix1 + row) + j) += *((matrix2 + row) + j);
    }
  }
}

/* train the neural network */
void train(double *inputs, double *targets) {
  /* calculate signals into hidden layer */
  double hidden_inputs[HNODES] = {0};
  mvp(HNODES, INODES, (double *)Wih, inputs, hidden_inputs);

  /* calculate the signals emerging from hidden layer */
  double hidden_outputs[HNODES] = {0};
  map(sigmoid, HNODES, hidden_inputs, hidden_outputs);

  /* calculate signals into final output layer */
  double final_inputs[ONODES] = {0};
  mvp(ONODES, HNODES, (double *)Who, hidden_outputs, final_inputs);

  /* calculate the signals emerging from final output layer */
  double final_outputs[ONODES] = {0};
  map(sigmoid, ONODES, final_inputs, final_outputs);

  /* output layer error is the (target - actual) */
  double output_errors[ONODES] = {0};
  vsubtv(targets, final_outputs, output_errors, ONODES);

  /* hidden layer error is the output_errors, split by weights, recombined at
   hidden nodes */
  double T[HNODES][ONODES] = {0};
  transpm((double *)Who, ONODES, HNODES, (double *)T);
  double hidden_errors[HNODES] = {0};
  mvp(HNODES, ONODES, (double *)T, output_errors, hidden_errors);

  /* update the weights for the links between the hidden and output layers */
  double tempho[ONODES] = {0};
  vmultv(output_errors, final_outputs, tempho, ONODES);
  oneminusv(final_outputs, ONODES);
  vmultv(tempho, final_outputs, tempho, ONODES);

  double dWho[ONODES][HNODES] = {0};
  vxv(ONODES, HNODES, tempho, hidden_outputs, (double *)dWho);
  smultm(ONODES, HNODES, (double *)dWho, LEARNING_RATE);
  mea(ONODES, HNODES, (double *)Who, (double *)dWho);

  /* update the weights for the links between the input and hidden layers */
  double tempih[HNODES] = {0};
  vmultv(hidden_errors, hidden_outputs, tempih, HNODES);
  oneminusv(hidden_outputs, HNODES);
  vmultv(tempih, hidden_outputs, tempih, HNODES);

  double dWih[HNODES][INODES] = {0};
  vxv(HNODES, INODES, tempih, inputs, (double *)dWih);
  smultm(HNODES, INODES, (double *)dWih, LEARNING_RATE);
  mea(HNODES, INODES, (double *)Wih, (double *)dWih);
}

char *getname(const char *matrix, int rows, int cols) {
  static char buffer[64];
  sprintf(buffer, "%s_%d_%d.csv", matrix, rows, cols);
  return buffer;
}

void tocsv(const char *csvfile, size_t rows, size_t cols,
           double arr[rows][cols]) {
  FILE *fd = fopen(csvfile, "w");
  if (fd == NULL) {
    LOG_ERROR("Cannot open file %s\n.", csvfile);
    exit(EXIT_FAILURE);
  }

  /* float can hold up to 7 decimal digits accurately while double can hold up
   * to 15. */
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      fprintf(fd, "%.15f%s", arr[i][j], (j < cols - 1 ? "," : ""));
    }
    fprintf(fd, "\n");
  }

  fclose(fd);
}

void fromcsv(const char *csvfile, size_t rows, size_t cols,
             double arr[rows][cols]) {
  const size_t MLEN = 40000;
  FILE *fd = fopen(csvfile, "r");
  if (fd == NULL) {
    LOG_ERROR("Cannot open file %s\n.", csvfile);
    exit(EXIT_FAILURE);
  }

  char buf[MLEN] = {0};
  for (size_t i = 0; i < rows; i++) {
    char *result = fgets(buf, MLEN, fd);
    if (result != NULL) {
      size_t j = 0;
      char *c = strtok(result, ",");
      while (c != NULL) {
        arr[i][j] = strtod(c, NULL);
        j++;
        c = strtok(NULL, ",");
      }
    } else {
      break;
    }
  }
  fclose(fd);
}
