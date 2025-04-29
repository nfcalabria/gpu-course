#include <stdio.h>

__device__ void function1() {
    // do something;
    return;
}

__device__ void function2() {
    // do something;
    return;
}

__global__ void helloDeadlock(){
    int index = threadIdx.x;
    printf("Hello world!!! thread Id: %d \n", index);

    if(index < 16) {
        function1();
        __syncthreads();
    } else {
        function2();
        __syncthreads();
    }
}

int main() {
    printf("Launch kernel!\n");
    helloDeadlock<<<1,32>>>();
    cudaDeviceSynchronize();
    printf("How I wish I were here\n");
    return 0;
}