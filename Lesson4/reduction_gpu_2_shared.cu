#include <stdio.h>
#include <random>
#include <chrono>

#define BLOCKSIZE 256
#define GRIDSIZE 256

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> timePoint;
typedef std::chrono::duration<double, std::milli> msInterval;

__global__ void reduce2(double *y, double *x, int N) {
    extern __shared__ double tsum[];

    int id = threadIdx.x;    
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int stride = gridDim.x*blockDim.x;

    tsum[id] = 0.;

    for(int k=tid; k<N; k+=stride){
        tsum[id] += x[k];        
    }
    __syncthreads();

    // Power of 2 reduction loop
    for(int k=blockDim.x/2; k>0; k/=2){
        if(id<k) tsum[id] += tsum[id+k];
        __syncthreads();
    }

    // Store one value per thread block
    if(id==0) y[blockIdx.x] = tsum[0];   
}

int main() {
    cudaDeviceReset();        
    int N = 2097152; // Any number, not necessarily a power of 2
    printf("reduce %d elements\n", N);
    double* h_a = new double[N]; // this time we use new, as in C++

    // Allocate device data
    double* d_a;
    cudaMalloc((void **) &d_a, sizeof(double)*N);

    // Allocate output vector
    double* d_out;
    cudaMalloc((void **) &d_out, sizeof(double)*GRIDSIZE);

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

    reduce2<<<GRIDSIZE, BLOCKSIZE, BLOCKSIZE*sizeof(double)>>>(
        d_out, d_a, N
    );

    reduce2<<<1, GRIDSIZE, GRIDSIZE*sizeof(double)>>>(
        d_a, d_out, GRIDSIZE
    );   

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