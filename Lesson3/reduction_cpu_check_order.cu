#include <stdio.h>
#include <random>

int main() {
    int N = 100; // Number of elements
    double* h_a = new double[N]; // this time we use new, as in C++

    // initalize random generator
    std::default_random_engine gen(42);
    std::uniform_real_distribution<double> fran(0.0, 1.0);

    for(int i = 0; i < N; i++){
        h_a[i] = fran(gen);
        // uncomment to show on screen the content of h_a
        // %a shows the Hexadecimal floating point representation
        // printf("h_a[%d] == %f %a\n", i, h_a[i], h_a[i]);
    }

    // Let's reduce it on the CPU
    double sum1 = 0.;
    for(int i = 0; i < N; i++) {
        sum1 += h_a[i];
    }

    printf("sum1 on CPU: %f %a\n", sum1, sum1);

    // Let's reduce it on the CPU performing sums in a different order
    double sum2 = 0.;
    for(int i = N-1; i >= 0; i--) {
        sum2 += h_a[i];
    }
    printf("sum2 on CPU: %f %a\n", sum2, sum2);

    // Let's try another order
    double sum3 = 0.;
    // even indexes first
    for(int i = 0; i < N; i=i+2) {
        sum3 += h_a[i];
    }
    // odd indexes now
    for(int i = 1; i < N; i=i+2) {
        sum3 += h_a[i];
    }
    printf("sum3 on CPU: %f %a\n", sum3, sum3);
    
    delete[] h_a; // remember to use delete[] for arrays instead of delete!!
    return 0;
}