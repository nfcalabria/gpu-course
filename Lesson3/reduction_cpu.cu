#include <stdio.h>
#include <random>

int main() {
    int N = 100; // Number of elements
    double* h_a = new double[N]; // this time we use new, as in C++

    // initalize random generator
    std::default_random_engine gen(42);
    std::uniform_real_distribution<double> fran(0.0, 1.0);

    for(unsigned int i = 0; i < N; i++){
        h_a[i] = fran(gen);
        // uncomment to show on screen the content of h_a
        // %a shows the Hexadecimal floating point representation
        // printf("h_a[%d] == %f %a\n", i, h_a[i], h_a[i]);
    }

    // Let's reduce it on the CPU
    double sum = 0.;
    for(unsigned int i = 0; i < N; i++) {
        sum += h_a[i];
    }

    printf("sum on CPU: %f %a\n", sum, sum);

    delete[] h_a; // remember to use delete[] for arrays instead of delete!!
    return 0;
}