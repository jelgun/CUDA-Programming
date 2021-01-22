#include "pgma_io.c"
#include <stdio.h>

// Simple histogram calucaltion
__global__
void histogram_simple(int * in, unsigned int * histo, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (i < size) {
        int val = in[i];
        atomicAdd(&(histo[val]), 1);
        i += stride;
    }
}

// Histogram calcuation with privatization
__global__
void histogram_private(int * in, unsigned int * histo, int size) {
    __shared__ unsigned int private_histo[256];
    private_histo[threadIdx.x] = 0;
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (i < size) {
        int val = in[i];
        atomicAdd(&(private_histo[val]), 1);
        i += stride;
    }

    __syncthreads();
    atomicAdd(&(histo[threadIdx.x]), private_histo[threadIdx.x]);
}


// Histogram calculation with privatization and aggregation
__global__
void histogram_prv_agg(int * in, unsigned int * histo, int size) {
    __shared__ unsigned int private_histo[256];
    private_histo[threadIdx.x] = 0;
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int prev_val = -1, acc = 0;
    while (i < size) {
        int cur_val = in[i];
        if (cur_val != prev_val) {
            if (prev_val != -1)
                atomicAdd(&(private_histo[prev_val]), acc);
            acc = 1;
            prev_val = cur_val;
        } else {
            acc++;
        }
        i += stride;
    }
    if (prev_val != -1)
        atomicAdd(&(private_histo[prev_val]), acc);
    __syncthreads();
    atomicAdd(&(histo[threadIdx.x]), private_histo[threadIdx.x]);
}

// Calulate cdf values with Brent Kung algorithm
__global__ 
void brent_kung_scan_kernel(unsigned int *X, int *Y, int size) {
    __const__ int BLOCK_SIZE = 256;
    __shared__ unsigned int XY[2*BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) 
        XY[threadIdx.x] = X[i];
    __syncthreads();
    for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index < 2*BLOCK_SIZE)
            XY[index] += XY[index - stride];
        __syncthreads();
    }

    for (unsigned int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index+stride < 2*BLOCK_SIZE) {
          XY[index + stride] += XY[index];
        }
    }
    __syncthreads();
    if (i < size) 
        Y[i] = XY[threadIdx.x];
}

__global__ 
void hist_equalizer(int *in, int *out, int *cdf, int cdf_min, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < size)
        out[i] = (int)((float)((float)(cdf[in[i]] - cdf_min) / (float)(size - cdf_min)) * 255.0);
        //printf("%d, %d, %d, %d, %d\n", out[i], in[i], cdf[in[i]], cdf_min, size);
}

int main(int argc, char ** argv) {
    cudaEvent_t start, stop;
  	cudaEventCreate(&start);
  	cudaEventCreate(&stop);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	
	// print device properties
	printf("Device name: %s\n", deviceProp.name);
	printf("Maximum number of blocks: %d\n", deviceProp.maxGridSize[0]);
	printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Maximum shared memory per block: %lu B\n", deviceProp.sharedMemPerBlock);

    char * inputImageFile;
    char * outputImageFile;
    int * hostInputImageData;
    int * hostOutputImageData;
    unsigned int * histogram = (unsigned int*)malloc(256 * sizeof(unsigned int));
    int * cdf = (int*)malloc(256 * sizeof(int));
    int imageWidth, imageHeight, maxVal;

    inputImageFile = argv[1];
    outputImageFile = argv[2];
    printf("Loading %s\n", inputImageFile);

    pgma_read(inputImageFile, &imageWidth, &imageHeight, &maxVal, &hostInputImageData);
    printf("%d %d %d\n", imageWidth, imageHeight, maxVal);

    int *d_in;
    unsigned int *d_out;
    int *d_img_out;
    int *d_cdf;
    int size = imageWidth*imageHeight;
    cudaMalloc((void**)&d_in, size * sizeof(int));
    cudaMalloc((void**)&d_img_out, size * sizeof(int));
    cudaMalloc((void**)&d_out, 256 * sizeof(unsigned int));
    cudaMalloc((void**)&d_cdf, 256 * sizeof(int));
	hostOutputImageData = (int *)malloc(size * sizeof(int));

    cudaMemcpy(d_in, hostInputImageData, size * sizeof(int), cudaMemcpyHostToDevice);
	dim3 gridSize = {(unsigned int)((size + 255)/256), 1, 1};
    dim3 blockSize = {256, 1, 1};

	cudaEventRecord(start);
    histogram_prv_agg<<<gridSize, blockSize>>>(d_in, d_out, size);
    brent_kung_scan_kernel<<<1, 256>>>(d_out, d_cdf, 256);
    cudaMemcpy(histogram, d_out, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cdf, d_cdf, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // find minimum non-zero cdf value
    int mn;
    for (int i = 0; i < 256; i++)
        if (cdf[i] != 0) {
            mn = cdf[i];
            break;
        }
    hist_equalizer<<<gridSize, blockSize>>>(d_in, d_img_out, d_cdf, mn, size);
    cudaEventRecord(stop);

    cudaMemcpy(hostOutputImageData, d_img_out, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free allocated memory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_cdf);
    printf("Histogram values:\n");
    for (int i = 0; i < 256; i++)
        printf("%d ", histogram[i]);
    printf("\nScan values:\n");
    for (int i = 0; i < 256; i++)
        printf("%d ", cdf[i]);
    printf("\n");
    
	cudaEventSynchronize(stop);
  	float milliseconds = 0;
  	cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f\n", milliseconds);
    maxVal = 0;
    for (int i = 0; i < size; i++)
        maxVal = max(maxVal, hostOutputImageData[i]);

    pgma_write (outputImageFile, "", imageWidth, imageHeight, maxVal, hostOutputImageData);

    return 0;
}
