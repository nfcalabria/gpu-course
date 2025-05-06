#include <stdio.h>

__constant__ cudaTextureObject_t c_texObj;
__constant__ int texSize;

__global__ void interpolateTexture(float * d_out) {
    int idx = blockDim.x*blockIdx.x + threadIdx.x;

    //d_out[idx] = tex1D<float>(c_texObj, idx + 0.5);
    d_out[idx] = tex1D<float>(c_texObj, idx + 0.5);    
    
}

__host__ void showArray(float * h_out, int size){
    printf("Show output:\n");
    for(int i = 0; i < size; i++){
        printf("Index: %d Value: %1.1f\n", i, h_out[i]);
    }
    printf("\n");
}

int main(){
    const int N = 6;

    // copy N to constant memory as texSize
    cudaMemcpyToSymbol(texSize, &N, sizeof(N));

    float h_data[N] = {1.,3.,2.,4.,5.,3.};

    // allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, N*sizeof(float), 0);    

    // Copy host array to cuda array in device memory
    // We use 2D functions to allocate a 1D array: 1D functions are DEPRECATED!    
    cudaMemcpy2DToArray(cuArray, 0, 0, h_data, N*sizeof(float), N*sizeof(float), 1, cudaMemcpyHostToDevice);

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    
    // FILTER MODE    
    texDesc.filterMode = cudaFilterModeLinear;
    // READ MODE
    texDesc.readMode = cudaReadModeElementType;
    // NORMALIZE COORDINATES
    int normalized = 0;
    texDesc.normalizedCoords = normalized;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    //Copy texObj to constant memory
    cudaMemcpyToSymbol(c_texObj, &texObj, sizeof(texObj));

    int blocks = 1;
    int threads = 8;

    // Allocate result of transformation in device memory
    float *d_output;
    cudaMalloc(&d_output, blocks*threads*sizeof(float));

    interpolateTexture<<<blocks, threads>>>(d_output);
    cudaDeviceSynchronize();

    //printf("%s",cudaGetErrorString(cudaGetLastError()));

    // copy results back from device memory

    float * h_output = new float[blocks*threads];
    cudaMemcpy(h_output, d_output, blocks*threads*sizeof(float), cudaMemcpyDeviceToHost);
    
    showArray(h_output, blocks*threads);

    cudaFree(d_output);
    //Dedicated cudaFree for Arrays!
    cudaFreeArray(cuArray);    
    delete[] h_output;

    return 0;
}