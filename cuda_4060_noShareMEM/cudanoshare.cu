#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <iostream>
#include <limits>

using namespace cv;
using namespace std;

// CUDA kernel for Gaussian blur
__global__ void gaussianBlurKernel(uchar3* input, uchar3* output, int width, int height, double* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = kernelSize / 2;

    if (x >= width || y >= height) return;

    double r = 0.0, g = 0.0, b = 0.0;

    for (int ky = -k; ky <= k; ky++) {
        for (int kx = -k; kx <= k; kx++) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);
            uchar3 pixel = input[ny * width + nx];
            double kval = kernel[(ky + k) * kernelSize + (kx + k)];
            r += pixel.x * kval;
            g += pixel.y * kval;
            b += pixel.z * kval;
        }
    }

    uchar3 out;
    out.x = static_cast<uchar>(r);
    out.y = static_cast<uchar>(g);
    out.z = static_cast<uchar>(b);
    output[y * width + x] = out;
}

// CUDA function for Gaussian blur
extern "C"
void gaussianBlurCUDA(unsigned char* inputPtr, unsigned char* outputPtr, int width, int height, double* kernel, int kernelSize) {
    uchar3* d_input;
    uchar3* d_output;
    double* d_kernel;

    size_t imgSize = width * height * sizeof(uchar3);
    size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(double);

    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMalloc(&d_kernel, kernelSizeBytes);

    cudaMemcpy(d_input, inputPtr, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSizeBytes, cudaMemcpyHostToDevice);
    float bestTime = std::numeric_limits<float>::max();
    int bestBx = 0, bestBy = 0;

    std::cout << "Benchmarking CUDA Gaussian blur with kernel size " << kernelSize << "..." << std::endl;

    for (int bx = 4; bx <= 32; bx += 4) {
        for (int by = 4; by <= 32; by += 4) {
            dim3 threadsPerBlock(bx, by);
            dim3 numBlocks((width + bx - 1) / bx, (height + by - 1) / by);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            gaussianBlurKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height, d_kernel, kernelSize);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);

            std::cout << "Block (" << bx << "x" << by << ") -> Time: " << milliseconds << " ms" << std::endl;

            if (milliseconds < bestTime) {
                bestTime = milliseconds;
                bestBx = bx;
                bestBy = by;
            }

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

    std::cout << "Best block size for Gaussian blur: (" << bestBx << "x" << bestBy << ") with time: " << bestTime << " ms" << std::endl;

    // Run the best configuration and store the result
    dim3 bestThreads(bestBx, bestBy);
    dim3 bestBlocks((width + bestBx - 1) / bestBx, (height + bestBy - 1) / bestBy);
    gaussianBlurKernel<<<bestBlocks, bestThreads>>>(d_input, d_output, width, height, d_kernel, kernelSize);
    cudaDeviceSynchronize();

    cudaMemcpy(outputPtr, d_output, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

// CUDA kernel for denoising (adaptive Gaussian blur)
__global__ void adaptiveDenoiseKernel(uchar3* input, uchar3* output, int width, int height, int neighborhoodSize, double factorRatio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pad = neighborhoodSize / 2;
    double r = 0.0, g = 0.0, b = 0.0;
    double weightSum = 0.0;

    for (int ky = -pad; ky <= pad; ky++) {
        for (int kx = -pad; kx <= pad; kx++) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);
            uchar3 pixel = input[ny * width + nx];

            double weight = exp(-(kx * kx + ky * ky) / (2 * factorRatio * factorRatio));
            r += pixel.x * weight;
            g += pixel.y * weight;
            b += pixel.z * weight;
            weightSum += weight;
        }
    }

    uchar3 out;
    out.x = static_cast<uchar>(r / weightSum);
    out.y = static_cast<uchar>(g / weightSum);
    out.z = static_cast<uchar>(b / weightSum);
    output[y * width + x] = out;
}

// CUDA function for adaptive denoising
extern "C"
void adaptiveDenoiseCUDA(unsigned char* inputPtr, unsigned char* outputPtr, int width, int height, int neighborhoodSize, double factorRatio) {
    uchar3* d_input;
    uchar3* d_output;

    size_t imgSize = width * height * sizeof(uchar3);

    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);

    cudaMemcpy(d_input, inputPtr, imgSize, cudaMemcpyHostToDevice);

    float bestTime = std::numeric_limits<float>::max();
    int bestBx = 0, bestBy = 0;

    std::cout << "Benchmarking CUDA adaptive denoise with neighborhood size " << neighborhoodSize << "..." << std::endl;

    for (int bx = 4; bx <= 32; bx += 4) {
        for (int by = 4; by <= 32; by += 4) {
            dim3 threadsPerBlock(bx, by);
            dim3 numBlocks((width + bx - 1) / bx, (height + by - 1) / by);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            adaptiveDenoiseKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height, neighborhoodSize, factorRatio);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);

            std::cout << "Block (" << bx << "x" << by << ") -> Time: " << milliseconds << " ms" << std::endl;

            if (milliseconds < bestTime) {
                bestTime = milliseconds;
                bestBx = bx;
                bestBy = by;
            }

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

    std::cout << "Best block size for adaptive denoise: (" << bestBx << "x" << bestBy << ") with time: " << bestTime << " ms" << std::endl;

    // Run the best configuration and store the result
    dim3 bestThreads(bestBx, bestBy);
    dim3 bestBlocks((width + bestBx - 1) / bestBx, (height + bestBy - 1) / bestBy);
    adaptiveDenoiseKernel<<<bestBlocks, bestThreads>>>(d_input, d_output, width, height, neighborhoodSize, factorRatio);
    cudaDeviceSynchronize();


    cudaMemcpy(outputPtr, d_output, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}