// MP 4
// 3D Convolution

#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define MASK_RADIUS 1
#define TILE_WIDTH 4

//@@ Define constant memory for device kernel here
__constant__ float mask[MASK_WIDTH*MASK_WIDTH*MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float inputTile[TILE_WIDTH+MASK_WIDTH-1][TILE_WIDTH+MASK_WIDTH-1][TILE_WIDTH+MASK_WIDTH-1];
  
  // Strategy 2
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  
  int dep_o = bz * TILE_WIDTH + tz;
  int row_o = by * TILE_WIDTH + ty;
  int col_o = bx * TILE_WIDTH + tx;
  
  int dep_i = dep_o - MASK_RADIUS;
  int row_i = row_o - MASK_RADIUS;
  int col_i = col_o - MASK_RADIUS;
  
  float value = 0.0;
  
  // taking care of boundaries
  if (dep_i >= 0 && dep_i < z_size && row_i >= 0 && row_i < y_size && col_i >= 0 && col_i < x_size)
    inputTile[tz][ty][tx] = input[dep_i * (y_size * x_size) + row_i * (x_size) + col_i];
  else
    inputTile[tz][ty][tx] = 0.0;
  
  __syncthreads();
  
  // calculating output
  if (tz < TILE_WIDTH && ty < TILE_WIDTH && tx < TILE_WIDTH)
  {
    for (int k = 0; k < MASK_WIDTH; k++)
      for (int i = 0; i < MASK_WIDTH; i++)
        for (int j = 0; j < MASK_WIDTH; j++)
          value += mask[k*MASK_WIDTH*MASK_WIDTH + i*MASK_WIDTH + j] * inputTile[k+tz][i+ty][j+tx];
    // writing output 
    if (dep_o < z_size && row_o < y_size && col_o < x_size)
      output[dep_o * (y_size * x_size) + row_o * (x_size) + col_o] = value; 
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int size = (inputLength-3)*sizeof(float);
  cudaMalloc((void**) &deviceInput,size);
  cudaMalloc((void**) &deviceOutput,size);
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpyToSymbol(mask,hostKernel,MASK_WIDTH*MASK_WIDTH*MASK_WIDTH*sizeof(float));
  cudaMemcpy(deviceInput, &hostInput[3], size, cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimBlock(TILE_WIDTH+MASK_WIDTH-1,TILE_WIDTH+MASK_WIDTH-1,TILE_WIDTH+MASK_WIDTH-1);
  dim3 dimGrid(ceil(x_size/(1.0*TILE_WIDTH)),ceil(y_size/(1.0*TILE_WIDTH)),ceil(z_size/(1.0*TILE_WIDTH)));

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid,dimBlock>>>(deviceInput,deviceOutput,z_size,y_size,x_size);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");
  cudaMemcpy(&hostOutput[3], deviceOutput, size, cudaMemcpyDeviceToHost);
  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
