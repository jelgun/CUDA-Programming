#include "wb.h"
#include <stdio.h>

void wbImage_save(const wbImage_t& image, const char* fName){
	std::ostringstream oss;
	oss << "P6\n" << "# Created for blurring output" << "\n" << image.width << " " << image.height << "\n" << image.colors << "\n";
	//oss << "P6\n" << "# Created by GIMP version 2.10.8 PNM plug-in" << "\n" << image.width << " " << image.height << "\n" << image.colors << "\n";

	std::string headerStr(oss.str());

	std::ofstream outFile(fName, std::ios::binary);
	outFile.write(headerStr.c_str(), headerStr.size());

	const int numElements = image.width * image.height * image.channels;

	unsigned char* rawData = new unsigned char[numElements];

	for (int i = 0; i < numElements; ++i)
	{
		rawData[i] = static_cast<unsigned char>(image.data[i] * wbInternal::kImageColorLimit + 0.5f);
	}

	outFile.write(reinterpret_cast<char*>(rawData), numElements);
	outFile.close();
	

	delete [] rawData;
}

__global__
void blurKernel(float * in, float * out, int w, int h) {
	__shared__ float s_in[18][18][3];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y*16 + ty;
	int col_o = blockIdx.x*16 + tx;

	int row_i = row_o - 1;
	int col_i = col_o - 1;
	if((row_i >= 0) && (row_i < h) && (col_i >= 0)  && (col_i < w)) {
		s_in[ty][tx][0] = in[row_i * w * 3 + col_i * 3];
		s_in[ty][tx][1] = in[row_i * w * 3 + col_i * 3 + 1];
		s_in[ty][tx][2] = in[row_i * w * 3 + col_i * 3 + 2];
	}
	__syncthreads();
	if (ty < 16 && tx < 16) {
		for (int channel = 0; channel < 3; channel++){
			float pixVal = 0;
			int pixels = 0;
			// Get the average of the surrounding 3 x 3 box
			for(int blurRow = -1; blurRow < 2; ++blurRow) {
				for(int blurCol = -1; blurCol < 2; ++blurCol) {
					int curRow = row_o + blurRow;
					int curCol = col_o + blurCol;
					// Verify we have a valid image pixel
					if(curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
						pixVal += s_in[curRow - 16 * blockIdx.y + 1][curCol - 16 * blockIdx.x + 1][channel];
						pixels++; // Keep track of number of pixels in the accumulated total
					}
				}
			}
			if (row_o < h && col_o < w) 
				out[row_o * w * 3 + col_o * 3 + channel] = (float)(pixVal / pixels);
		}
	}

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
    wbImage_t inputImage;
    wbImage_t outputImage;

    float * hostInputImageData;
    float * hostOutputImageData;

    inputImageFile = argv[1];
    outputImageFile = argv[2];
    printf("Loading %s\n", inputImageFile);
    inputImage = wbImport(inputImageFile);
    hostInputImageData = wbImage_getData(inputImage);

    int imageWidth = wbImage_getWidth(inputImage);
    int imageHeight = wbImage_getHeight(inputImage);
    int imageChannels = wbImage_getChannels(inputImage);

    hostInputImageData = wbImage_getData(inputImage);
    printf("%d %d %d\n", imageWidth, imageHeight, imageChannels);

	/*YOUR CODE HERE*/

	float *d_in, *d_out;
	long size = imageWidth*imageHeight*imageChannels*sizeof(float);
	cudaMalloc((void**)&d_in, size);
	cudaMalloc((void**)&d_out, size);
	hostOutputImageData = (float *)malloc(size);

	cudaMemcpy(d_in, hostInputImageData, size, cudaMemcpyHostToDevice);

	dim3 gridSize = {(unsigned int)((imageWidth + 15)/16), (unsigned int)((imageHeight + 15)/16)};
	dim3 blockSize = {18, 18};
	cudaEventRecord(start);
    blurKernel<<<gridSize, blockSize>>>(d_in, d_out, imageWidth, imageHeight);
	cudaEventRecord(stop);
	
	cudaMemcpy(hostOutputImageData, d_out, size, cudaMemcpyDeviceToHost);
	// free allocated memory
	cudaFree(d_in);
	cudaFree(d_out);

	cudaEventSynchronize(stop);
  	float milliseconds = 0;
  	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f\n", milliseconds);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    outputImage.data = hostOutputImageData;
    wbImage_save(outputImage, outputImageFile);    

    return 0;
}
