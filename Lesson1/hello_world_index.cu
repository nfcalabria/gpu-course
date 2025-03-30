#include <stdio.h>

__global__ void helloGPU(){
    int index = threadIdx.x;
    printf("Hello world!!! thread Id: %d \n", index);
}

int main() {
    printf("Launch kernel!\n");
    helloGPU<<<1,1000>>>();
    cudaDeviceSynchronize();
    return 0;
}
