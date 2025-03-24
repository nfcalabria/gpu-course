#include <stdio.h>

__global__ void helloGPU(){
    int index = threadIdx.x;
    printf("Hello world!!! thread Id: %d \n", index);
}

int main() {
    printf("Launch kernel!\n");
    // Play with the number of threads in a a block!
    // Put a number greater than 1024 and see what happens!
    helloGPU<<<1,1025>>>();
    cudaDeviceSynchronize();
    return 0;
}