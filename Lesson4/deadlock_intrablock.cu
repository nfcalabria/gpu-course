#include <stdio.h>

__global__ void deadlock(int gsync, int dolock){
    __shared__ int lock;
    if(threadIdx.x == 0) lock = 0;
    __syncthreads(); // normal syncthreads

    if(threadIdx.x < gsync) { // group A branch
        __syncthreads();
        if (threadIdx.x == 0) lock = 1;
    }
    else if (threadIdx.x < 2*gsync) { // group B branch
        __syncthreads();        
    }

    // group C executes only this part
    if(dolock) while(lock != 1); 
}

int main() {    
    int warps = 3; //group A, B and C
    int blocks = 1;
    int gsync = 32;
    int dolock = 1;
    printf("Launch kernel!\n");
    deadlock<<<blocks, warps*32>>>(gsync, dolock);
    cudaDeviceSynchronize();
    printf("done\n");
    return 0;
}
    