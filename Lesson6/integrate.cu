#include <stdio.h>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> timePoint;
typedef std::chrono::duration<double, std::milli> msInterval;

__global__ void stencil2D(double *d_a, double *d_b, int nx, int ny) {
    auto idx = [&nx] (int y, int x) { return y*nx + x; };
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    // Return if out of bound
    if(x < 1 || y < 1 || x >= nx-1 || y >= ny-1) return;

    d_b[idx(y,x)] = 0.25*(d_a[idx(y,x+1)] + d_a[idx(y,x-1)] + d_a[idx(y+1,x)] + d_a[idx(y-1,x)]);
    
}

int stencil2D_host(double * a, double * b, int nx, int ny) {
    auto idx = [&nx] (int y, int x) { return y*nx + x; };
    // omit edges
    for(int y=1;y<ny-1;y++) {
        for(int x=1;x<nx-1;x++){
                b[idx(y,x)] = 0.25*( a[idx(y,x+1)] + a[idx(y,x-1)] + a[idx(y+1,x)] + a[idx(y-1,x)] );
        }
    }
    return 0;
}

void showOnScreen(double * out, int nx) {
    auto idx = [&nx] (int y, int x) { return y*nx + x; };

    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++){
            printf("%f ", out[idx(i, j)]);
        }
        printf("\n");
    }
}

int main() {
    int nx = 1024;
    int ny = 1024;
    int iter_host = 1000;
    int iter_gpu = 1000;
    int size = nx*ny;

    //host grid
    double *a = new double[size]();
    double *b = new double[size]();
    //device grid
    double * d_a, * d_b;
    cudaMalloc(&d_a, sizeof(double)*size);
    cudaMalloc(&d_b, sizeof(double)*size);

    auto idx = [&nx](int y, int x) {return y*nx + x; };
    // Set boundaries
    for(int y=0; y<ny; y++) a[idx(y,0)] = a[idx(y, nx-1)] = 1.0;
    // Corner adjustment
    a[idx(0,0)] = a[idx(0,nx-1)] = a[idx(ny-1,0)] = a[idx(ny-1, nx-1)] = 0.5;

    //Copy to device
    cudaMemcpy(d_a, a, sizeof(double)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(double)*size, cudaMemcpyHostToDevice);

    timePoint start = Clock::now();

    for(int k=0; k<iter_host/2; k++){
        stencil2D_host(a, b, nx, ny);
        stencil2D_host(b, a, nx, ny); // Double buffering
    }

    timePoint stop = Clock::now();
    msInterval interval = stop - start;
    printf("CPU elapsed time: %f ms \n", interval.count());
    printf("CPU content:\n");
    showOnScreen(a, nx);

    dim3 threads = {16,16,1};
    dim3 blocks = {
        (nx+threads.x-1)/threads.x, (ny+threads.y-1)/threads.y,1
    };

    start = Clock::now();

    for(int k=0;k<iter_gpu/2;k++){
        stencil2D<<<blocks, threads>>>(a, b, nx, ny);
        stencil2D<<<blocks, threads>>>(b, a, nx, ny); // Double buffering
    }
    cudaDeviceSynchronize();

    stop = Clock::now();
    interval = stop - start;
    printf("GPU elapsed time: %f ms \n", interval.count());
    
    cudaMemcpy(a, d_a, sizeof(double)*size, cudaMemcpyDeviceToHost);
    printf("GPU content:\n");
    showOnScreen(a, nx);
        
    delete[] a;
    delete[] b;
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}