#include <stdio.h>

#define ERRCHK(error) {                                                                         \
    const cudaError_t e = error;                                                                \
    if(e!=cudaSuccess) {                                                                        \
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));             \
      exit(0);                                                                                  \
    }                                                                                           \
   }

__global__ void sumVectors(int* a, int* b, int* c, int maxN) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;    
    if(i >= maxN) return; // It's better to be safe than to be sorry
    c[i] = a[i] + b[i];
}

int main() {

    cudaDeviceReset();

    //dynamically allocated arrays
    const int N = 10; // Size of arrays

    int *h_a, *h_b, *h_c, *h_cgpu; // h_ stands for HOST
    int *d_a, *d_b, *d_c; // d_stands for DEVICE

    h_a = (int*) malloc(sizeof(int)*N); //a
    h_b = (int*) malloc(sizeof(int)*N); //b
    h_c = (int*) malloc(sizeof(int)*N); //c
    h_cgpu = (int*) malloc(sizeof(int)*N); //c

    // Device allocations
    ERRCHK(cudaMalloc(&d_a, sizeof(int)*N));
    ERRCHK(cudaMalloc(&d_b, sizeof(int)*N));
    ERRCHK(cudaMalloc(&d_c, sizeof(int)*N));

    printf("ALLOCATED!\n");
    printf("initialization:\n");

    // INITIALIZE a and b
    for(unsigned int i = 0; i < N; i++) {
        h_a[i] = i*2;       //even int numbers
        h_b[i] = i*2 + 1;   //odd int numbers
    }

    // calculate sum
    for(unsigned int i = 0; i < N; i++) {
        h_c[i] = h_a[i] + h_b[i];
    }

    // print results
    for(unsigned int i = 0; i < N; i++) {
        printf("idx: %d, %d + %d = %d\n", i, h_a[i], h_b[i], h_c[i]);
    }

    // Now let's do it on the GPU
    ERRCHK(cudaMemcpy(d_a, h_a, sizeof(int)*N, cudaMemcpyHostToDevice));
    ERRCHK(cudaMemcpy(d_b, h_b, sizeof(int)*N, cudaMemcpyHostToDevice));    

    sumVectors<<<1, N>>>(d_a, d_b, d_c, N);
    ERRCHK(cudaDeviceSynchronize()); // This is how we check the kernels

    // Overwrite h_c with device results
    ERRCHK(cudaMemcpy(h_cgpu, d_c, sizeof(int)*N, cudaMemcpyDeviceToHost));

    // print GPU results
    for(unsigned int i = 0; i < N; i++) {
        printf("idx: %d, %d + %d = %d\n", i, h_a[i], h_b[i], h_cgpu[i]);
    }
    
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_cgpu);

    // free device memory
    ERRCHK(cudaFree(d_a));
    ERRCHK(cudaFree(d_b));
    ERRCHK(cudaFree(d_c));
    return 0;
}