// include any headers
#include <iostream>
#include <math.h>

// define constants
#define BLOCK_SIZE 1024
#define SIZE 1000000

// function to apply histogram on arrays
__global__ void histo_kernel(int *buffer, int size, int *bin) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // stride is total number of threads
    int stride = blockDim.x * gridDim.x;

    // bin calc
    while(i < size){
        atomicAdd(&bin[buffer[i]], 1);
        i += stride;
    }
}

int main(void) {
    // allocate input and histogram arrays
    int *input, *d_bin;
    int bin[20] = {0};
    cudaMallocManaged(&input, SIZE*sizeof(int));
    cudaMallocManaged(&d_bin, 20*sizeof(int));

    // initialize input array on the host
    for(int i=0; i<SIZE; i++){
        input[i] = rand() % 20;
    }

    int gridSize = (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Zero out the histogram bins on the device
    cudaMemset(d_bin, 0, 10 * sizeof(int));

    // Run the histogram kernel
    histo_kernel<<<gridSize, BLOCK_SIZE>>>(input, SIZE, d_bin);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(bin, d_bin, 20 * sizeof(int), cudaMemcpyDeviceToHost);

    // check for error
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    } else {
        std::cout << "Kernel completed successfully!" << std::endl;
    }

    // Print the histograms
    /*
    printf("input array: \n");
    for(int i = 0; i < SIZE; i++) {
        printf("%d \n", input[i]);
    }
    */

    std::cout << "\nHistogram for numbers:" << std::endl;
    for (int i = 0; i < 19; i++) {
        std::cout << "Bin " << i << ": " << bin[i] << std::endl;
    }

  // free memory
    cudaFree(input);
    cudaFree(bin);
    return 0;
}
