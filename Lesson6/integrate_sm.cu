#include <stdio.h>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::time_point<Clock> timePoint;
typedef std::chrono::duration<double, std::milli> msInterval;

template <int Nx, int Ny> __global__ void stencil2D_sm(double * d_a, double * d_b, int nx, int ny)
{
    __shared__ double s[Ny][Nx];

    auto idx = [&nx] (int y, int x) { return y*nx + x; };
    // tile origins
    int x0 = ( blockDim.x - 2) * blockIdx.x;
    int y0 = ( blockDim.y - 2) * blockIdx.y;
    // Array index
    int xa = x0 + threadIdx.x;
    int ya = y0 + threadIdx.y;
    // Tile index
    int xs = threadIdx.x;
    int ys = threadIdx.y;
    if( xa >= nx || ya >= ny ) return; // return if out of boundaries

    s[ys][xs] = d_a[idx(ya, xa)];
    __syncthreads();
    // Shared memory is ready

    // Inside array?
    if(xa < 1 || ya < 1 || xa >= nx - 1 || ya >= ny - 1) return;
    // Inside tile?
    if(xs < 1 || ys < 1 || xs >= Nx - 1 || ys >= Ny - 1) return;

    d_b[idx(ya,xa)] = 0.25*(s[ys][xs+1] + s[ys][xs-1] + s[ys+1][xs] + s[ys-1][xs]);
    
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

    for(int i = nx-10; i < nx; i++) {
        for(int j = nx-10; j < nx; j++){
            printf("%f ", out[idx(i, j)]);
        }
        printf("\n");
    }
}

int main() {
    cudaDeviceReset();
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

    const int Nx = 16;
    const int Ny = 16;
    const int halo = 1;

    dim3 threads = {16,16,1};
    dim3 blocks = {
        (nx+(threads.x-2*halo)-1)/(threads.x-2*halo), 
        (ny+(threads.y-2*halo)-1)/(threads.y-2*halo),
        1
    };

    start = Clock::now();

    for(int k=0;k<iter_gpu/2;k++){
        stencil2D_sm<Nx,Ny><<<blocks, threads>>>(d_a, d_b, nx, ny);
        stencil2D_sm<Nx,Ny><<<blocks, threads>>>(d_b, d_a, nx, ny); // Double buffering
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