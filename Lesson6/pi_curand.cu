#include <stdio.h>
#include <chrono>
#include <random>
#include <curand.h>

// COMPILE WITH -lcurand FLAG!!!

#define PI 3.14159265358979323846

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> timePoint;
typedef std::chrono::duration<double, std::milli> msInterval;

void sum_part(float *rnum, int points, long long &pisum) {
    unsigned int sum = 0;
    for(int i = 0; i < points; i++){
        float x = rnum[i*2];
        float y = rnum[i*2+1];
        if(x*x + y*y < 1.0f) sum++;
    }
    pisum += sum;
}

int main() {
    std::random_device rd;    

    int points = 1000000;
    int passes = 100;
    long long pisum = 0;
    unsigned int seed = rd();
    
    // buffers
    float* rdm = new float[points*2];
    float * d_rdm;
    cudaMalloc(&d_rdm, sizeof(float)*2*points);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    timePoint start = Clock::now();

    for(int k = 0; k < passes; k++) {
        curandGenerateUniform(gen, d_rdm, points*2);
        // Copy random numbers to host
        cudaMemcpy(rdm, d_rdm, sizeof(float)*2*points, cudaMemcpyDeviceToHost);
        
        // Use them with host function for integration
        sum_part(rdm, points, pisum);
    }

    timePoint stop = Clock::now();
    msInterval interval = stop - start;

    double pi = 4.0 * (double) pisum / ((double) points * (double) passes);
    printf("CPU elapsed time: %f ms \n", interval.count());

    double frac_error = 1000000.0*(pi - (PI))/(PI); //ppm
    long long ntot = (long long)passes*(long long)points;

    printf("pi = %10.8f err %.1f, ntot %lld\n",pi, frac_error, ntot);

    delete[] rdm;
    cudaFree(d_rdm);
    return 0;
}