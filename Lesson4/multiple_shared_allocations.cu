#include <stdio.h>

#define BLOCKSIZE 32
#define GRIDSIZE 1

__global__ void kernel_static() {
    __shared__ int data_1[32];
    __shared__ int data_2[32];

    int idx = threadIdx.x;

    // let's initialize data_2
    data_2[idx] = 42;

    // let's write idx to data_1
    data_1[idx] = idx;

    __syncthreads();

    // let's print data
    printf("STATIC data_1[%d] = %d, data_2[%d] = %d\n", idx, data_1[idx], idx, data_2[idx]);
}

__global__ void kernel_dynamic_WRONG() {
    extern __shared__ int data_1[];
    extern __shared__ int data_2[];

    int idx = threadIdx.x;

    // let's initialize data_2
    data_2[idx] = 42;

    // let's write idx to data_1
    data_1[idx] = idx;

    __syncthreads();

    // let's print data
    printf("DYNAMIC WRONG data_1[%d] = %d, data_2[%d] = %d\n", idx, data_1[idx], idx, data_2[idx]);
}

__global__ void kernel_dynamic_RIGHT() {
    extern __shared__ int data[]; // Single array
       
    // Split it using pointers.
    int * data_1 = (int*) &data[0];
    int * data_2 = (int*) &data[32];

    int idx = threadIdx.x;
    // let's initialize data_2
    data_2[idx] = 42;

    // let's write idx to data_1
    data_1[idx] = idx;

    __syncthreads();

    // let's print data
    printf("DYNAMIC RIGHT data_1[%d] = %d, data_2[%d] = %d\n", idx, data_1[idx], idx, data_2[idx]);
}

int main() {
    cudaDeviceReset();
    
    kernel_static<<<GRIDSIZE, BLOCKSIZE>>>();

    kernel_dynamic_WRONG<<<GRIDSIZE, BLOCKSIZE, sizeof(int)*(32+32)>>>();
    
    kernel_dynamic_RIGHT<<<GRIDSIZE, BLOCKSIZE, sizeof(int)*(32+32)>>>();
    
    cudaDeviceSynchronize();
    return 0;
}