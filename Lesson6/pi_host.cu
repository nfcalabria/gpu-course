#include <stdio.h>
#include <chrono>
#include <random>

#define PI 3.14159265358979323846

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> timePoint;
typedef std::chrono::duration<double, std::milli> msInterval;

int main() {
    std::random_device rd;
    int points = 1000000;
    int passes = 100;
    unsigned int seed = rd();

    std::default_random_engine gen(seed);
    std::uniform_real_distribution<double> ddist(0.0, 1.0);

    long long pisum = 0;
    timePoint start = Clock::now();

    for(int n = 0; n < passes; n++){
        int subtot = 0;
        for(int k = 0; k < points; k++){
            double x = ddist(gen);
            double y = ddist(gen);
            if(x*x + y*y < 1.0) subtot ++; // inside circle            
        }
        pisum += subtot;
    }    

    timePoint stop = Clock::now();
    msInterval interval = stop - start;

    double pi = 4.0 * (double) pisum / ((double) points * (double) passes);
    printf("CPU elapsed time: %f ms \n", interval.count());

    double frac_error = 1000000.0*(pi - (PI))/(PI); //ppm
    long long ntot = (long long)passes*(long long)points;

    printf("pi = %10.8f err %.1f, ntot %lld\n",pi, frac_error, ntot);

    return 0;
}