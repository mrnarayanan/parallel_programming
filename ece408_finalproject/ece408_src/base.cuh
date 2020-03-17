#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 32

namespace mxnet
{
namespace op
{

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

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
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int tz = threadIdx.z + blockIdx.z * blockDim.z;

    // Local compute variable
    float conv_val = 0.0;

    // Convolution Code
    for (int m = 0; m < M; m++){    // Loop through images
        conv_val = 0.0; // Reset val for new image
        for (int c = 0; c < C; c++){    // Loop through input features
            // Loop through 2D convolution kernel/filter
            for (int kx = 0; kx < K; kx++){
                for (int ky = 0; ky < K; ky++){
                    if ((tz < B) && (ty + ky < H) && (tx + kx < W)){ // if dimensions are within input bounds and M
                        conv_val += k4d(m, c, ky, kx) * x4d(tz, c, ty+ky, tx+kx);   // do convolution
                    }
                }
            }
        }
        if ((tx < W_out) && (ty < H_out) && (tz < B)){  // if thread is within output bounds
            y4d(tz, m, ty, tx) = conv_val;  // set convolution result
        }
    }


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

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(ceil((1.0*W_OUT)/(1.0*TILE_WIDTH)), ceil((1.0*H_OUT)/(1.0*TILE_WIDTH)), B);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim, 0>>>(y.dptr_,x.dptr_,k.dptr_, B,M,C,H,W,K);

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

#undef TILE_WIDTH
#endif
