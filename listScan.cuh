// MP 5.2
// List Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void add(float *input, float *output, int len) 
{
  int bx = blockIdx.x;
  int index = bx * blockDim.x + threadIdx.x;
  
  if (index < len && bx > 0)
    output[index] += input[bx-1];
}

__global__ void scan(float *input, float *output, int len) 
{
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  // reduction step
  
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
    output[idx] = T[tx];
  if (idx + bd < len)
    output[idx+bd] = T[bd+tx];
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *hostIntermed;
  float *deviceIntermed;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);
  
  int numSections = numElements/(BLOCK_SIZE*2);
  wbLog(TRACE, "The number of sections is ",
        numSections);
  hostIntermed = (float *) malloc(numSections * sizeof(float));

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceIntermed, numSections * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here  
  int primary_dim = ceil(numElements/(float) BLOCK_SIZE/2);
  int secondary_dim = ceil(numSections/(float) (BLOCK_SIZE*2));

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the device
  scan<<<primary_dim,BLOCK_SIZE>>>(deviceInput,deviceOutput,numElements);
  cudaDeviceSynchronize();

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");
  
  if (numSections > 0)
  {
    for (int i = 1; i <= numSections; i++)
      hostIntermed[i-1] = hostOutput[BLOCK_SIZE*2*i-1];
    wbCheck(cudaMemcpy(deviceIntermed, hostIntermed, numSections * sizeof(float), cudaMemcpyHostToDevice));
    scan<<<secondary_dim,BLOCK_SIZE>>>(deviceIntermed,deviceIntermed,numSections);
    cudaDeviceSynchronize();
    
    wbCheck(cudaMemcpy(deviceOutput, hostOutput, numElements * sizeof(float), cudaMemcpyHostToDevice));
    add<<<primary_dim,BLOCK_SIZE*2>>>(deviceIntermed,deviceOutput,numElements);
    cudaDeviceSynchronize();
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                       cudaMemcpyDeviceToHost));
  }
  
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
