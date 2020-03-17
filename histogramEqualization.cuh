// MP 6
// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 128
#define TILE_WIDTH 32

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ insert code here
__global__ void cast_start(float *input, unsigned char *output, int width, int height, int channels)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (ii < width * height * channels)
    output[ii] = (unsigned char) (255 * input[ii]);
}

__global__ void to_grayscale(unsigned char *input, unsigned char *output, int width, int height)
{
  int jj = blockIdx.y * blockDim.y + threadIdx.y; // height
  int ii = blockIdx.x * blockDim.x + threadIdx.x; // width 
 
  if (ii < width && jj < height)
  {
    int idx = jj * width + ii;
    unsigned char r = input[3*idx];
    unsigned char g = input[3*idx+1];
    unsigned char b = input[3*idx+2];
    output[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void compute_histogram(unsigned char *input, unsigned int *output, int width, int height)
{
  __shared__ unsigned int histo[HISTOGRAM_LENGTH];
  if (threadIdx.x < HISTOGRAM_LENGTH)
    histo[threadIdx.x] = 0;
  __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  while (i < width * height)
  {
    atomicAdd(&(histo[input[i]]), 1);
    i += stride;
  }
  __syncthreads();
  if (threadIdx.x < HISTOGRAM_LENGTH)
    atomicAdd(&(output[threadIdx.x]), histo[threadIdx.x]);
}

__global__ void compute_cdf(unsigned int *input, float *output, int len, int width, int height)
{
  __shared__ float T[2*BLOCK_SIZE];
  
  int tx = threadIdx.x;
  int bd = blockDim.x;
  int idx = 2 * bd * blockIdx.x + tx;
  
  // load input
  if (idx < len)
    T[tx] = input[idx];
  else
    T[tx] = 0.0;
  if (idx + bd < len)
    T[bd + tx] = input[idx + bd];
  else
    T[bd + tx] = 0.0;
  
  int stride = 1;
  while (stride <= bd)
  {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index < 2*BLOCK_SIZE && (index-stride) >= 0)
      T[index] += T[index-stride];
    stride *= 2;
  }
  
  stride = BLOCK_SIZE/2;
  while (stride > 0)
  {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if ((index+stride) < 2*BLOCK_SIZE)
      T[index+stride] += T[index];
    stride /= 2;
  }
  
  // write output
  __syncthreads();
  if (idx < len)
    output[idx] = T[tx]/((float)width*height);
  if (idx + bd < len)
    output[idx+bd] = T[bd+tx]/((float)width*height);
}

__global__ void equalize(float *input, unsigned char *output, int width, int height)
{
  int jj = blockIdx.y * blockDim.y + threadIdx.y; // height
  int ii = blockIdx.x * blockDim.x + threadIdx.x; // width 
  
  if (jj < height && ii < width)
  {
    int idx = jj * width + ii;
    float eq = 255.0 * (input[output[3*idx]] - input[0]) / (1.0 - input[0]);
    float clamp = min(max(eq,0.0),255.0);
    output[3*idx] = (unsigned char) clamp;
    
    eq = 255.0 * (input[output[3*idx+1]] - input[0]) / (1.0 - input[0]);
    clamp = min(max(eq,0.0),255.0);
    output[3*idx+1] = (unsigned char) clamp;
    
    eq = 255.0 * (input[output[3*idx+2]] - input[0]) / (1.0 - input[0]);
    clamp = min(max(eq,0.0),255.0);
    output[3*idx+2] = (unsigned char) clamp;
    __syncthreads();
  }
}

__global__ void cast_end(unsigned char *input, float *output, int width, int height, int channels)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;
 
  if (ii < width * height * channels)
    output[ii] = (float) (input[ii]/255.0);
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  unsigned char *ucharImage;
  unsigned char *grayImage;
  unsigned int *histogram;
  float *cdf;
  float *deviceImage;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  hostInputImageData = wbImage_getData(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");
  
  wbLog(TRACE, "The image dimensions are (width x height x channels): ",
        imageWidth, " x ", imageHeight, " x ", imageChannels);

  //@@ insert code here
  wbTime_start(GPU, "Allocating GPU Memory");
  wbCheck(cudaMalloc((void **) &deviceImage, imageWidth * imageHeight * imageChannels * sizeof(float)));
  wbCheck(cudaMalloc((void **) &ucharImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **) &grayImage, imageWidth * imageHeight * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **) &histogram, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(cudaMalloc((void **) &cdf, HISTOGRAM_LENGTH * sizeof(float)));
  wbCheck(cudaMemset(histogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned char)));
  wbTime_stop(GPU, "Allocating GPU Memory");
  
  wbTime_start(GPU, "Copying mem to GPU");
  wbCheck(cudaMemcpy(deviceImage, hostInputImageData, 
                     imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying mem to GPU");
  
  wbTime_start(GPU, "GPU computations");
  dim3 dimGrid;
  dim3 dimBlock;
  int fullsize = imageWidth * imageHeight * imageChannels;
  
  // cast_start
  dimGrid = dim3(ceil(fullsize/(float)TILE_WIDTH),1,1);
  dimBlock = dim3(TILE_WIDTH,1,1);
  
  cast_start<<<dimGrid,dimBlock>>>(deviceImage, ucharImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  
  // to_grayscale
  dimGrid = dim3(ceil(imageWidth/(float)TILE_WIDTH),ceil(imageHeight/(float)TILE_WIDTH),1);
  dimBlock = dim3(TILE_WIDTH,TILE_WIDTH,1);
  
  to_grayscale<<<dimGrid,dimBlock>>>(ucharImage, grayImage, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  // compute_histogram
  dimGrid = dim3(ceil((imageWidth*imageHeight)/(float)HISTOGRAM_LENGTH),1,1);
  dimBlock = dim3(HISTOGRAM_LENGTH,1,1);
  
  compute_histogram<<<dimGrid,dimBlock>>>(grayImage, histogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  // compute_cdf
  dimGrid = dim3(1,1,1);
  dimBlock = dim3(ceil(HISTOGRAM_LENGTH/2.0),1,1);
  
  compute_cdf<<<dimGrid,dimBlock>>>(histogram, cdf, HISTOGRAM_LENGTH, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  // equalize
  dimGrid = dim3(ceil(imageWidth/(float)TILE_WIDTH),ceil(imageHeight/(float)TILE_WIDTH),1);
  dimBlock = dim3(TILE_WIDTH,TILE_WIDTH,1);
  
  equalize<<<dimGrid,dimBlock>>>(cdf, ucharImage, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  // cast_end
  dimGrid = dim3(ceil(fullsize/(float)TILE_WIDTH),1,1);
  dimBlock = dim3(TILE_WIDTH,1,1);
  
  cast_end<<<dimGrid,dimBlock>>>(ucharImage, deviceImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  
  wbTime_stop(GPU, "GPU computations");
  
  wbTime_start(GPU, "Copying mem from GPU");
  cudaMemcpy(hostOutputImageData, deviceImage, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(GPU, "Copying mem from GPU");
  
  wbImage_setData(outputImage,hostOutputImageData);
  wbSolution(args, outputImage);

  //@@ insert code here
  wbTime_start(GPU, "Free GPU Memory");
  wbCheck(cudaFree(ucharImage));
  wbCheck(cudaFree(grayImage));
  wbCheck(cudaFree(histogram));
  wbCheck(cudaFree(cdf));
  wbTime_stop(GPU, "Free GPU Memory");

  return 0;
}
