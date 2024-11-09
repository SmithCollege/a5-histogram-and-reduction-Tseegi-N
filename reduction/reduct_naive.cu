// include any headers
#include <iostream>
#include <math.h>

// define constants
#define BLOCK_SIZE 256
#define SIZE 100

// function to apply reduction on arrays
__global__ void reduct(int *in, int *out, int n){
    // initialize shared mem
    __shared__ int shared[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // store memory
    if(idx <n){
        shared[tid] = in[idx];
    }
    else{
        shared[tid] = 0;
    }
    __syncthreads();

    // compute
    for (unsigned int stride = 1; stride <= blockDim.x;  stride *= 2) {
        if (tid % (2 * stride) == 0 && tid + stride < blockDim.x){
            shared[tid]+= shared[tid+stride];
        }
        __syncthreads();
    }

    // store results back to output
    if(tid == 0){
        out[blockIdx.x] = shared[0];
    }
}

int main(void) {
    // allocate input and output arrays
    int *input, *output;
    cudaMallocManaged(&input, SIZE*sizeof(int));
    cudaMallocManaged(&output, SIZE*sizeof(int));
    // initialize input array on the host
    for(int i=0; i<SIZE; i++){
        input[i] = 1;
    }

    // initialize temp pointers and size
    int elmtSize = SIZE;
    int *temp_in = input;
    int *temp_out = output;

   //
    while(elmtSize > 1){
        int gridSize = (elmtSize + BLOCK_SIZE - 1)/BLOCK_SIZE;
        reduct<<<gridSize, BLOCK_SIZE>>>(temp_in, temp_out, elmtSize);
        cudaDeviceSynchronize();

       // update values
       elmtSize = gridSize;
       int *temp = temp_in;
       temp_in = temp_out;
       temp_out = temp;
    }

    // check for error
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    } else {
        std::cout << "Kernel completed successfully!" << std::endl;
    }

    std::cout << "Result: " << output[0] << std::endl;

    // print out input and output array
    std::cout << "Input array:" << std::endl;
    for (int i = 0; i < SIZE; i++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Output array:" << std::endl;
    for (int i = 0; i < SIZE; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

  // free memory
    cudaFree(input);
    cudaFree(output);
    return 0;
}
