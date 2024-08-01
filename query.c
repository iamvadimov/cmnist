/* File query.c */

#include "common.h"
#include <inttypes.h> // PRIu64
#include <stdlib.h>

#define TEST_FILE "mnist_dataset/mnist_test.csv"
#define TEST_ROWS 10000
#define DATA_COLS (1 + 28 * 28)

int idxmax(double *vec, int length) {
  double res = vec[0];
  int idx = 0;
  for (int i = 1; i < length; i++) {
    if (res < vec[i]) {
      res = vec[i];
      idx = i;
    }
  }
  return idx;
}

void viewres(int *vec, int length) {
  for (int i = 0; i < length; i++) {
    printf("%4d\t", vec[i]);
  }
  putchar('\n');
}

int main() {
  uint64_t bytes = (TEST_ROWS * DATA_COLS) * sizeof(double);
  double *ptrtd = (double *)malloc(bytes);
  /* Check if the memory has been successfully allocated by malloc or not */
  if (ptrtd == NULL) {
    fprintf(stderr, "Memory not allocated.");
    exit(EXIT_FAILURE);
  }

  printf("Testing data: %s\n", TEST_FILE);
  readcsv(TEST_FILE, TEST_ROWS, DATA_COLS, (double *)ptrtd);
  /* PRIu64 is a format specifier, introduced in C99, for printing uint64_t */
  printf("Size of testing data %" PRIu64 " Bytes: %s\n", bytes,
         human_size(bytes));

  const char *wihname = getname("wih", HNODES, INODES);
  fromcsv(wihname, HNODES, INODES, Wih);
  bytes = (HNODES * INODES) * sizeof(double); // sizeof(Wih);
  printf("Size of Wih %" PRIu64 " Bytes: %s\n", bytes, human_size(bytes));

  const char *whoname = getname("who", ONODES, HNODES);
  fromcsv(whoname, ONODES, HNODES, Who);
  bytes = (ONODES * HNODES) * sizeof(double); // sizeof(Who);
  printf("Size of Who %" PRIu64 " Bytes: %s\n", bytes, human_size(bytes));

  /* scorecard for how well the network performs, initially 0 */
  int success[ONODES] = {0};
  int fail[ONODES] = {0};
  int total = 0;
  double(*TD)[TEST_ROWS][DATA_COLS] = (void *)ptrtd;
  for (int i = 0; i < TEST_ROWS; i++) {
    double *ptrd = (*TD)[i];
    int label = (int)(*ptrd);
    double targets[ONODES] = {[0 ... ONODES - 1] = 0.01};
    targets[label] = 0.99;
    ptrd++; // skip 0-element; now ptrd is a pointer to inputs

    double final_outputs[ONODES] = {0};
    query(ptrd, final_outputs);
    int idx = idxmax(final_outputs, ONODES);
    if (idx == label) {
      success[label]++;
      total++;
    } else {
      fail[label]++;
    }
  }

  /* the performance score, the fraction of correct answers */
  printf("Performance score: %.2f\n", (total / (double)TEST_ROWS));
  viewres(success, ONODES);
  viewres(fail, ONODES);

  free(ptrtd);
  return 0;
}
