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
  int* output = malloc(sizeof(int) * SIZE);

  // initialize inputs
  for (int i = 0; i < SIZE; i++) {
    if(i == 50){
      input[i] = i+100;
    }
    else{
      input[i] = i;
    }
  }

  //initialize output
  output[0] = input[0];

  // Get time
  double time0, time1;
  time0 = get_clock();

  // addition
  for (int i = 1; i < SIZE; i++) {
   output[i] = output[i-1] + input[i];
  }

  // max
  /*
  for (int i = 1; i < SIZE; i++) {
    if (output[i-1] < input[i]) {
      output[i] = input[i];
    }
    else{
      output[i] = output[i-1];
    }
  }
  */

  // Final time
  time1 = get_clock();
  printf("time: %f seconds\n", (time1-time0));

  // check results
  /*
  for (int i = 0; i < SIZE; i++) {
    //printf("%d ", input[i]);
    printf("%d ", output[i]);
  }
  printf("\n");
  */

  // free mem
  free(input);
  free(output);

  return 0;
}
