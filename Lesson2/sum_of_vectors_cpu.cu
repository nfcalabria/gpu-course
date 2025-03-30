#include <stdio.h>

int main() {

    //dynamically allocated arrays
    const int N = 10; // Size of arrays

    int *h_a, *h_b, *h_c; // h_ stands for HOST

    h_a = (int*) malloc(sizeof(int)*N); //a
    h_b = (int*) malloc(sizeof(int)*N); //b
    h_c = (int*) malloc(sizeof(int)*N); //c

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
    
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}