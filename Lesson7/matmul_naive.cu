#include <stdio.h>
#include <random>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> timePoint;
typedef std::chrono::duration<double, std::milli> msInterval;

__global__ void gpumult(double* A, double* B, double* C, int Arow, int Acol, int Bcol)
{
    int tx = blockIdx.x*blockDim.x + threadIdx.x; // j
    int ty = blockIdx.y*blockDim.y + threadIdx.y; // i

    if(ty >= Arow || tx >= Bcol) return;
    C[ty*Bcol+tx] = 0.0;
    for(int k=0; k < Acol; k++){
        C[ty*Bcol+tx] += A[ty*Bcol+k]*B[k*Bcol+tx];
    }
}

void hostmult(double* A, double* B, double* C, int Arow, int Acol, int Bcol)
{ 
    // Ci,j: i spans Arow, j spans Bcol
    // Acol == Brow       
    for(int i = 0; i < Arow; i++){        
        for(int j = 0; j < Bcol; j++){
            C[i*Bcol+j] = 0.;
            for(int k=0;k<Acol;k++)C[i*Bcol+j] += A[i*Acol+k]*B[k*Bcol+j];            
        }
    }    
}

void showMat(double * M, int Mcol){
    const int N = 10;

    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            printf("%f ", M[i*Mcol + j] );
        }
        printf("\n");
    }
}

int main(){
    int Arow = 1024; // Can edit
    int Acol = Arow; // Can edit

    int Brow = Acol;
    int Bcol = Brow; // Can edit

    int Crow = Arow;
    int Ccol = Bcol;

    int Asize = Arow*Acol;
    int Bsize = Brow*Bcol;
    int Csize = Brow*Bcol;

    // allocate buffers
    double *A, *B, *C;
    A = new double[Asize];
    B = new double[Bsize];
    C = new double[Csize];

    // Device buffers
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(double)*Asize);
    cudaMalloc(&d_B, sizeof(double)*Bsize);
    cudaMalloc(&d_C, sizeof(double)*Csize);

    // Initialize A and B with random numbers
    std::default_random_engine gen(42);
    std::uniform_real_distribution<float> fran(0.0, 1.0);
    for(int k = 0; k<Asize; k++) A[k] = fran(gen);
    for(int k = 0; k<Bsize; k++) B[k] = fran(gen);

    // Copy to GPU buffers
    cudaMemcpy(d_A, A, sizeof(double)*Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(double)*Bsize, cudaMemcpyHostToDevice);

    timePoint start;
    timePoint stop;
    msInterval interval;


    // CPU REDUCTION
    start = Clock::now();
    hostmult(A,B,C,Arow,Acol,Bcol);
    stop = Clock::now();
    interval = stop - start;

    showMat(C, Ccol);
    printf("CPU ELAPSED TIME: %f ms\n", interval.count());

    // GPU REDUCTION

    uint tilex = 32;
    uint tiley = 8;
    dim3 threads = {tilex, tiley, 1};
    dim3 blocks = { (Bcol+threads.x-1)/ threads.x, (Arow+threads.y-1)/threads.y, 1};

    start = Clock::now();
    gpumult<<<blocks,threads>>>(d_A,d_B,d_C,Arow,Acol,Bcol);
    cudaDeviceSynchronize();
    stop = Clock::now();
    interval = stop - start;

    cudaMemcpy(C, d_C, sizeof(double)*Csize, cudaMemcpyDeviceToHost);
    showMat(C, Ccol);
    printf("GPU ELAPSED TIME: %f ms\n", interval.count());

    // Cleanup

    delete[] A;
    delete[] B;
    delete[] C;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}