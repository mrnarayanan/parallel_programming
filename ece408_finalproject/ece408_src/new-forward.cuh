
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH_LARGE 24
#define TILE_WIDTH_SMALL 16

namespace mxnet
{
namespace op
{

__global__ void forward_kernel_large(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    __shared__ float tileMatA[TILE_WIDTH_LARGE][TILE_WIDTH_LARGE];
    __shared__ float tileMatB[TILE_WIDTH_LARGE][TILE_WIDTH_LARGE];

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
// i3 = image, i2 = feature, i1 = height/row/y, i0 = width/column/x
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Get the thread index
    int b = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH_LARGE + ty;
    int col = blockIdx.x * TILE_WIDTH_LARGE + tx;
    int numMatACols = C*K*K; // = numMatBRows

    // Local compute variable
    float conv_val = 0.0;

    // Convolution Code
    int num_iter = ceil(numMatACols/(1.0*TILE_WIDTH_LARGE));

    for (int i = 0; i < num_iter; i++)
    {
        int temp_col = i * TILE_WIDTH_LARGE + tx;
        int temp_row = i * TILE_WIDTH_LARGE + ty;

        // filter tensor
        int k_m = row;
        int k_c = temp_col / (K*K);
        int k_h = (temp_col % (K*K)) / K;
        int k_w = (temp_col % (K*K)) % K;

        // input tensor
        int x_b = b;
        int x_c = temp_row / (K*K);
        int x_p = (temp_row % (K*K)) / K;
        int x_q = (temp_row % (K*K)) % K;
        int x_h = col / W_out;
        int x_w = col % W_out;

        if (temp_col < numMatACols && row < M)
            tileMatA[ty][tx] = k4d(k_m, k_c, k_h, k_w);
        else
            tileMatA[ty][tx] = 0.0;

        if (temp_row < numMatACols && col < H_out * W_out)
            tileMatB[ty][tx] = x4d(x_b, x_c, x_h + x_p, x_w + x_q);
        else
            tileMatB[ty][tx] = 0.0;

        __syncthreads();

        for (int q = 0; q < TILE_WIDTH_LARGE; q++)
            conv_val += tileMatA[ty][q] * tileMatB[q][tx];

        __syncthreads();
    }

    // output tensor
    int y_b = b;
    int y_m = row;
    int y_h = col / W_out;
    int y_w = col % W_out;

    if (row < M && col < W_out * H_out)
        y4d(y_b, y_m, y_h, y_w) = conv_val;

#undef y4d
#undef x4d
#undef k4d
}

__global__ void forward_kernel_small(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    __shared__ float tileMatA[TILE_WIDTH_SMALL][TILE_WIDTH_SMALL];
    __shared__ float tileMatB[TILE_WIDTH_SMALL][TILE_WIDTH_SMALL];

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
// i3 = image, i2 = feature, i1 = height/row/y, i0 = width/column/x
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Get the thread index
    int b = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH_SMALL + ty;
    int col = blockIdx.x * TILE_WIDTH_SMALL + tx;
    int numMatACols = C*K*K; // = numMatBRows

    // Local compute variable
    float conv_val = 0.0;

    // Convolution Code
    int num_iter = ceil(numMatACols/(1.0*TILE_WIDTH_SMALL));

    for (int i = 0; i < num_iter; i++)
    {
        int temp_col = i * TILE_WIDTH_SMALL + tx;
        int temp_row = i * TILE_WIDTH_SMALL + ty;

        // filter tensor
        int k_m = row;
        int k_c = temp_col / (K*K);
        int k_h = (temp_col % (K*K)) / K;
        int k_w = (temp_col % (K*K)) % K;

        // input tensor
        int x_b = b;
        int x_c = temp_row / (K*K);
        int x_p = (temp_row % (K*K)) / K;
        int x_q = (temp_row % (K*K)) % K;
        int x_h = col / W_out;
        int x_w = col % W_out;

        if (temp_col < numMatACols && row < M)
            tileMatA[ty][tx] = k4d(k_m, k_c, k_h, k_w);
        else
            tileMatA[ty][tx] = 0.0;

        if (temp_row < numMatACols && col < H_out * W_out)
            tileMatB[ty][tx] = x4d(x_b, x_c, x_h + x_p, x_w + x_q);
        else
            tileMatB[ty][tx] = 0.0;

        __syncthreads();

        for (int q = 0; q < TILE_WIDTH_SMALL; q++)
            conv_val += tileMatA[ty][q] * tileMatB[q][tx];

        __syncthreads();
    }

    // output tensor
    int y_b = b;
    int y_m = row;
    int y_h = col / W_out;
    int y_w = col % W_out;

    if (row < M && col < W_out * H_out)
        y4d(y_b, y_m, y_h, y_w) = conv_val;

#undef y4d
#undef x4d
#undef k4d
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &k)
{

    // Use mxnet's CHECK_EQ to do assertions.

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];  // number of images
    const int M = y.shape_[1];  // output feature map
    const int C = x.shape_[1];  // input feature map
    const int H = x.shape_[2];  // input height
    const int W = x.shape_[3];  // input width
    const int K = k.shape_[3];  // conv filter dim
    const int H_OUT = H - K + 1;// output height
    const int W_OUT = W - K + 1;// output width

    if (C==1){
        dim3 blockDim(TILE_WIDTH_SMALL, TILE_WIDTH_SMALL, 1);
        dim3 gridDim(ceil((H_OUT*W_OUT)/(1.0*TILE_WIDTH_SMALL)), ceil((M)/(1.0*TILE_WIDTH_SMALL)), B);

        // Call the kernel
        forward_kernel_small<<<gridDim, blockDim, 0>>>(y.dptr_,x.dptr_,k.dptr_, B,M,C,H,W,K);
    }
    else{
        dim3 blockDim(TILE_WIDTH_LARGE, TILE_WIDTH_LARGE, 1);
        dim3 gridDim(ceil((H_OUT*W_OUT)/(1.0*TILE_WIDTH_LARGE)), ceil((M)/(1.0*TILE_WIDTH_LARGE)), B);

        // Call the kernel
        forward_kernel_large<<<gridDim, blockDim, 0>>>(y.dptr_,x.dptr_,k.dptr_, B,M,C,H,W,K);
    }

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &k)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#undef TILE_WIDTH_SMALL
#undef TILE_WIDTH_LARGE
#endif
