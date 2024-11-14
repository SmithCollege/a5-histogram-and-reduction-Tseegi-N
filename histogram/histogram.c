#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#define SIZE 1000000

// clock func
double  get_clock() {
  struct timeval tv;
  int ok;
  ok = gettimeofday(&tv, NULL);
  if (ok<0) {
    //printf('gettimeofday error\n');
  }
  return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int main() {
  // allocate memory
  int* input = malloc(sizeof(int) * SIZE);
  int hist[20] = {0};

  // initialize inputs
  for (int i = 0; i < SIZE; i++) {
    input[i] = rand() % 20;
  }

  // Get time
  double time0, time1;
  time0 = get_clock();

  // histogram
  for (int i = 0; i < SIZE; i++) {
     hist[input[i]]++;
  }

  // Final time
  time1 = get_clock();
  printf("time: %f seconds\n", (time1-time0));

  // check results
  printf("Histogram for numbers 0 through 19:\n");
  for (int i = 0; i < 20; i++) {
    printf("Number %d: %d occurrences\n", i, hist[i]);
  }
  printf("\n");


  // free mem
  free(input);
  //free(hist);

  return 0;
}
