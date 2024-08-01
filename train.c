/* File train.c */
#include "common.h"
#include <inttypes.h> /* PRIu64 */
#include <stdlib.h>
#include <time.h>

#define DATA_COLS (1 + 28 * 28)
#define TRAIN_FILE "mnist_dataset/mnist_train.csv"
#define TRAIN_ROWS 60000

/* clang -Wall train.c common.c -lm -o nn */

int main(void) {
  clock_t begin = clock();

  uint64_t bytes = (TRAIN_ROWS * DATA_COLS) * sizeof(double);
  double *ptrtd = (double *)malloc(bytes);
  /* Check if the memory has been successfully allocated by malloc or not */
  if (ptrtd == NULL) {
    fprintf(stderr, "Memory not allocated.");
    exit(EXIT_FAILURE);
  }
  printf("Training data: %s\n", TRAIN_FILE);
  readcsv(TRAIN_FILE, TRAIN_ROWS, DATA_COLS, (double *)ptrtd);
  /* PRIu64 is a format specifier, introduced in C99, for printing uint64_t */
  printf("Size of training data %" PRIu64 " Bytes: %s\n", bytes,
         human_size(bytes));

  fill_random((double *)Wih, HNODES, INODES);
  fill_random((double *)Who, ONODES, HNODES);

  /* train the neural network */
  /* epochs is the number of times the training data set is used for training */
  int epochs = 5; 
  double(*TD)[TRAIN_ROWS][DATA_COLS] = (void *)ptrtd;
  for (int e = 0; e < epochs; e++) {
    printf("epoch %d\n", e);
    for (int i = 0; i < TRAIN_ROWS; i++) {
      double *ptrd = (*TD)[i];
      //   printf("ptrd = %g\n", *ptrd);
      double targets[ONODES] = {[0 ... ONODES - 1] = 0.01};
      targets[(int)(*ptrd)] = 0.99;
      //   printv(targets, ONODES);
      ptrd++; // skip 0-element; now ptrd is a pointer to inputs

      train(ptrd, targets);
    }
  }

  free(ptrtd);

  const char *wihname = getname("wih", HNODES, INODES);
  tocsv(wihname, HNODES, INODES, Wih);
  const char *whoname = getname("who", ONODES, HNODES);
  tocsv(whoname, ONODES, HNODES, Who);

  double time_spent = (double)(clock() - begin) / CLOCKS_PER_SEC;
  printf("Time spent:\t%.2f seconds.\n", time_spent);

  return 0;
}


