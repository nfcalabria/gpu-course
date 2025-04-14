#include <stdio.h>
#include <random>
#include <chrono>

#define BLOCKSIZE 64
#define GRIDSIZE 128

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> timePoint;
typedef std::chrono::duration<double, std::milli> msInterval;

__global__ void reduce1(double *x, int N) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    double tsum = 0.;
    for(int k=tid; k<N; k += gridDim.x*blockDim.x) {        
        // store partial sums in first 
        // BLOCKSIZE * GRIDSIZE element of x
        tsum += x[k]; 
    }
    x[tid] = tsum;    
}

int main() {
    cudaDeviceReset();    
    int N = 2097152; // Any number, not necessarily a power of 2
    printf("reduce %d elements\n", N);
    double* h_a = new double[N]; // this time we use new, as in C++

    // Allocate device data
    double* d_a;
    cudaMalloc((void **) &d_a, sizeof(double)*N);

    // initalize random generator
    std::default_random_engine gen(42);
    std::uniform_real_distribution<double> fran(0.0, 1.0);

    for(unsigned int i = 0; i < N; i++){
        h_a[i] = fran(gen);
        // uncomment to show on screen the content of h_a
        // %a shows the Hexadecimal floating point representation
        // printf("h_a[%d] == %f %a\n", i, h_a[i], h_a[i]);
    }

    // Copy vector to GPU
    cudaMemcpy(d_a, h_a, sizeof(double)*N, cudaMemcpyHostToDevice);

    timePoint start = Clock::now();

    // Let's reduce it on the CPU
    double sum = 0.;
    for(unsigned int i = 0; i < N; i++) {
        sum += h_a[i];
    }

    timePoint stop = Clock::now();
    msInterval interval = stop - start;    

    printf("sum on CPU: %f %a\n", sum, sum);
    printf("CPU elapsed time: %f ms \n", interval.count());

    start = Clock::now();

    reduce1<<<GRIDSIZE, BLOCKSIZE>>>(d_a, N);
    reduce1<<<1, BLOCKSIZE>>>(d_a, BLOCKSIZE*GRIDSIZE);
    reduce1<<<1,1>>>(d_a, BLOCKSIZE);
    // Synchronize before time measurement to be sure that all threads are done
    cudaDeviceSynchronize();   

    stop = Clock::now();
    interval = stop - start;    

    // sum is in d_a[0], let's copy it from the device to the host variable sum
    double sum_gpu;
    cudaMemcpy(&sum_gpu, d_a, sizeof(double), cudaMemcpyDeviceToHost);
    
    printf("sum on GPU: %f %a\n", sum_gpu, sum_gpu);    
    printf("GPU elapsed time: %f ms \n", interval.count());

    // Clean memory
    cudaFree(d_a);
    delete[] h_a; // remember to use delete[] for arrays instead of delete!!
    return 0;
}