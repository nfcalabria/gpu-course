#include <stdio.h>

__global__ void helloGPU(){
    printf("Hello world!!!\n");
}

int main() {
    printf("Launch kernel!\n");
    helloGPU<<<1,10>>>(); //Asynchronous launch
    cudaDeviceSynchronize();
    return 0;
}
