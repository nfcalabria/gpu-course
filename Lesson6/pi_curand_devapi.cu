#include <stdio.h>
#include <chrono>
#include <random>
#include <curand_kernel.h>

// COMPILE WITH -lcurand FLAG!!!

#define PI 3.14159265358979323846
#define REDUCTION_BLOCKSIZE 64
#define REDUCTION_GRIDSIZE 128

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> timePoint;
typedef std::chrono::duration<double, std::milli> msInterval;

template <typename S> __global__ void init_generator(long long seed, S *states)
{
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    curand_init(seed, id, 0, &states[id]);
}

template <typename S> __global__ void piG(float *tsum, S *states, int points)
{
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    S state = states[id];
    float sum = 0.0f;
    for(int i = 0; i < points; i++){
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if(x*x + y*y < 1.0f) sum++;
    }
    tsum[id] += sum;
    states[id] = state;
}

// reduce1 from Lesson4
__global__ void reduce1(float *x, int N) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    float tsum = 0.0f;
    for(int k=tid; k<N; k += gridDim.x*blockDim.x) {        
        // store partial sums in first 
        // BLOCKSIZE * GRIDSIZE element of x
        tsum += x[k]; 
    }
    x[tid] = tsum;    
}

int main() {
    std::random_device rd;
    
    int shift = 18;
    long long seed = rd();
    int blocks = 2048;
    int threads = 1024;

    // ntot = 2^shift
    long long ntot = (long long) 1 << shift;

    int size = threads*blocks;
    int nthread = (ntot+size-1)/size;
    ntot = (long long) nthread*size;
    float *d_tsum;
    cudaMalloc(&d_tsum, sizeof(float)*size);
    cudaMemset(d_tsum, 0, sizeof(float)*size);

    curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState)*size);    

    timePoint start = Clock::now();

    init_generator<<<blocks,threads>>>(seed, d_state);

    piG<<<blocks,threads>>>(d_tsum, d_state, nthread);

    timePoint stop = Clock::now();
    msInterval interval = stop - start;

    // We need to reduce t_sum!

    reduce1<<<REDUCTION_GRIDSIZE, REDUCTION_BLOCKSIZE>>>(d_tsum, size);
    reduce1<<<1, REDUCTION_BLOCKSIZE>>>(d_tsum, REDUCTION_BLOCKSIZE*REDUCTION_GRIDSIZE);
    reduce1<<<1,1>>>(d_tsum, REDUCTION_BLOCKSIZE);
    // Synchronize before time measurement to be sure that all threads are done
    cudaDeviceSynchronize();
    
    double pisum;
    cudaMemcpy(&pisum, d_tsum, sizeof(double), cudaMemcpyDeviceToHost);

    double pi = 4.0 * (double) pisum / ((double) ntot);
    printf("CPU elapsed time: %f ms \n", interval.count());

    double frac_error = 1000000.0*(pi - (PI))/(PI); //ppm

    printf("pi = %10.8f err %.1f, ntot %lld\n",pi, frac_error, ntot);


    return 0;
}