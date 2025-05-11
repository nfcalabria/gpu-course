#include <stdio.h>
#include <random>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> timePoint;
typedef std::chrono::duration<double, std::milli> msInterval;

template <int TS>__global__ void gpumult_tiled(double* __restrict A, double* __restrict B, double* __restrict C, int Arow, int Acol, int Bcol)
{
    __shared__ double Atile[TS][TS];
    __shared__ double Btile[TS][TS];

    int tx = threadIdx.x; // tile col index j
    int ty = threadIdx.y; // tile row index i
    int ocx = blockDim.x*blockIdx.x; // tile x origin in C
    int ocy = blockDim.y*blockIdx.y; // tile y origin in C

    int ax = tx;
    int ay = ocy+ty;
    int bx = ocx+tx;
    int by = ty;

    double csum = 0.;
    for(int t=0; t<gridDim.x; t++) {
        Atile[ty][tx] = A[ay*Acol + ax];
        Btile[ty][tx] = B[by*Bcol + bx];
        __syncthreads();
        for(int k=0; k<TS; k++) csum += Atile[ty][k] * Btile[k][tx];
        __syncthreads();
        
        ax += TS;
        by += TS;
    }
    C[ay*Bcol+bx] = csum;
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

    const uint tilex = 32;    
    dim3 threads = {tilex, tilex, 1};
    dim3 blocks = { (Bcol+threads.x-1)/ threads.x, (Arow+threads.y-1)/threads.y, 1};

    start = Clock::now();
    gpumult_tiled<tilex><<<blocks,threads>>>(d_A,d_B,d_C,Arow,Acol,Bcol);
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