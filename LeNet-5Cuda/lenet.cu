/*
The MIT License(MIT)
Copyright(c) 2016 Fan Wen Jie

Permission is hereby granted, free of charge, to any person obtaining a copy
of this softwareand associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// Source: https://github.com/fan-wenjie/LeNet-5

#include "lenet.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cstdio>


#define CUDAMALLOC_CHECK(ptr, size)									 \
{																	 \
	if (cudaMalloc((void**)&ptr, size) != cudaSuccess) {		     \
		fprintf(stderr, "ERROR: cudaMalloc for %s failed!\n", #ptr); \
		return -1;													 \
	}																 \
}																	 \

int CudaInit()
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "ERROR: cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		return -1;
	}
	return 0;
}

int CudaDeInit()
{
	cudaError_t cudaStatus;
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "ERROR: cudaDeviceReset failed!");
		return -1;
	}
	return 0;
}

int LeNetCudaAlloc(LeNet5Cuda* lenet5)
{
	// Allocate GPU buffers for LeNet5 data.
	CUDAMALLOC_CHECK(lenet5->weight0_1, INPUT * LAYER1 * LENGTH_KERNEL * LENGTH_KERNEL * sizeof(double));
	CUDAMALLOC_CHECK(lenet5->weight2_3, LAYER2 * LAYER3 * LENGTH_KERNEL * LENGTH_KERNEL * sizeof(double));
	CUDAMALLOC_CHECK(lenet5->weight4_5, LAYER4 * LAYER5 * LENGTH_KERNEL * LENGTH_KERNEL * sizeof(double));
	CUDAMALLOC_CHECK(lenet5->weight5_6, (LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5) * OUTPUT * sizeof(double));
	CUDAMALLOC_CHECK(lenet5->bias0_1, LAYER1 * sizeof(double));
	CUDAMALLOC_CHECK(lenet5->bias2_3, LAYER3 * sizeof(double));
	CUDAMALLOC_CHECK(lenet5->bias4_5, LAYER5 * sizeof(double));
	CUDAMALLOC_CHECK(lenet5->bias5_6, OUTPUT * sizeof(double));

	return 0;
}
int LeNetCudaFree(LeNet5Cuda* lenet5)
{
	cudaFree(lenet5->weight0_1);
	cudaFree(lenet5->weight2_3);
	cudaFree(lenet5->weight4_5);
	cudaFree(lenet5->weight5_6);
	cudaFree(lenet5->bias0_1);
	cudaFree(lenet5->bias2_3);
	cudaFree(lenet5->bias4_5);
	cudaFree(lenet5->bias5_6);
	return 0;
}

int FeatureCudaAlloc(FeatureCuda* feature)
{
	CUDAMALLOC_CHECK(feature->input, INPUT * LENGTH_FEATURE0 * LENGTH_FEATURE0 * sizeof(double));
	CUDAMALLOC_CHECK(feature->layer1, LAYER1 * LENGTH_FEATURE1 * LENGTH_FEATURE1 * sizeof(double));
	CUDAMALLOC_CHECK(feature->layer2, LAYER2 * LENGTH_FEATURE2 * LENGTH_FEATURE2 * sizeof(double));
	CUDAMALLOC_CHECK(feature->layer3, LAYER3 * LENGTH_FEATURE3 * LENGTH_FEATURE3 * sizeof(double));
	CUDAMALLOC_CHECK(feature->layer4, LAYER4 * LENGTH_FEATURE4 * LENGTH_FEATURE4 * sizeof(double));
	CUDAMALLOC_CHECK(feature->layer5, LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5 * sizeof(double));
	CUDAMALLOC_CHECK(feature->output, OUTPUT*sizeof(double));

	return 0;
}

int FeatureCudaFree(FeatureCuda* feature)
{
	cudaFree(feature->input);
	cudaFree(feature->layer1);
	cudaFree(feature->layer2);
	cudaFree(feature->layer3);
	cudaFree(feature->layer4);
	cudaFree(feature->layer5);
	cudaFree(feature->output);
	return 0;
}

#define CUDAMEMCPY_CHECK(src, dest, bytes, type)												      \
{																								      \
	cudaError_t err;																				  \
	if ((err = cudaMemcpy(dest, src, bytes, type)) != cudaSuccess)								      \
		fprintf(stderr, "ERROR(%i): cudaMemCpy %s from %s to %s failed!\n", err, #type, #src, #dest); \
}


#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

#define CONVOLUTE_VALID(input,output,weight)											\
{																						\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
}

#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}

__global__ void ForwardConvoluteKernel(const double const* input, double* output, const double const* weight, const int inputFeatures, const int inputW, const int inputH, const double const* bias)
{
	int outFeature = blockIdx.z;
	int outputFeatures = gridDim.z;
	int threadW = blockIdx.x * (LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1) + threadIdx.x;
	int threadH = blockIdx.y * (LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1) + threadIdx.y;

	double acc = 0;

	__shared__ double inTile[LENGTH_KERNEL_TILE][LENGTH_KERNEL_TILE];
	__shared__ double weightFilter[LENGTH_KERNEL][LENGTH_KERNEL];
	
	for (int inFeature = 0; inFeature < inputFeatures; inFeature++)
	{
		if ((threadW < inputW) && (threadH < inputH))
		{
			inTile[threadIdx.y][threadIdx.x] = input[inFeature * inputH * inputW +
													 threadH * inputW +
													 threadW];
		}
		if ((threadIdx.x < LENGTH_KERNEL) && (threadIdx.y < LENGTH_KERNEL))
		{
			weightFilter[threadIdx.y][threadIdx.x] = weight[inFeature * outputFeatures * LENGTH_KERNEL * LENGTH_KERNEL +
															outFeature * LENGTH_KERNEL * LENGTH_KERNEL +
															threadIdx.y * LENGTH_KERNEL +
															threadIdx.x];
		}
		__syncthreads();

		if ((threadIdx.x < (LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1)) && (threadIdx.y < (LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1)) &&
			(threadH < (inputH - LENGTH_KERNEL + 1)) && (threadW < (inputW - LENGTH_KERNEL + 1)))
		{
			for (int p = 0; p < LENGTH_KERNEL; p++)
			{
				for (int q = 0; q < LENGTH_KERNEL; q++)
				{
					acc += inTile[threadIdx.y + p][threadIdx.x + q] * weightFilter[p][q];
				}
			}
		}
		__syncthreads();
	}

	if ((threadIdx.x < (LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1)) && (threadIdx.y < (LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1)) &&
		(threadH < (inputH - LENGTH_KERNEL + 1)) && (threadW < (inputW - LENGTH_KERNEL + 1)))
	{
		acc += bias[outFeature];
		output[outFeature * (inputH - LENGTH_KERNEL + 1) * (inputW - LENGTH_KERNEL + 1) + threadH * (inputW - LENGTH_KERNEL + 1) + threadW] = acc * (acc > 0);
	}
}

// Similar functionality as the code in Figure 16.4 of the textbook
void ConvolutionForward(double* input, double* output, double* weight, double* bias, const int inputFeatures, const int outputFeatures, const int inputWidth, const int inputHeight)					\
{
	/*Blocks are fixed sized tiles to allow for any size of input*/
	dim3 block(LENGTH_KERNEL_TILE, LENGTH_KERNEL_TILE, 1);
	unsigned int tilesW = ceil((float)(inputWidth - LENGTH_KERNEL + 1) / (float)(LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1));
	unsigned int tilesH = ceil((float)(inputHeight - LENGTH_KERNEL + 1) / (float)(LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1));
	dim3 grid(tilesW, tilesH, outputFeatures);
	ForwardConvoluteKernel <<< grid, block >>> (input, output, weight, inputFeatures, inputWidth, inputHeight, bias);
}

__global__ void ReverseConvoluteKernel(const double const* input, double* output, const double const* weight, const int inputFeatures, const int inputW, const int inputH)
{
	int outFeature = blockIdx.z;
	int outputFeatures = gridDim.z;
	int threadW = blockIdx.x * (LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1) + threadIdx.x;
	int threadH = blockIdx.y * (LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1) + threadIdx.y;

	int readX = threadW - LENGTH_KERNEL + 1;
	int readY = threadH - LENGTH_KERNEL + 1;

	double acc = 0;

	__shared__ double inTile[LENGTH_KERNEL_TILE][LENGTH_KERNEL_TILE];
	__shared__ double weightFilter[LENGTH_KERNEL][LENGTH_KERNEL];

	for (int inFeature = 0; inFeature < inputFeatures; inFeature++)
	{
		if ((readX < inputW) && (readY < inputH))
		{
			if ((readX >= 0) && (readY >= 0))
			{
				inTile[threadIdx.y][threadIdx.x] = input[inFeature * inputH * inputW +
														 readY * inputW +
														 readX];
			}
			else
			{
				inTile[threadIdx.y][threadIdx.x] = 0;
			}
		}
		if ((threadIdx.x < LENGTH_KERNEL) && (threadIdx.y < LENGTH_KERNEL))
		{
			weightFilter[LENGTH_KERNEL - 1 - threadIdx.y][LENGTH_KERNEL - 1 - threadIdx.x] = weight[outFeature * inputFeatures * LENGTH_KERNEL * LENGTH_KERNEL +
																									inFeature * LENGTH_KERNEL * LENGTH_KERNEL +
																									threadIdx.y * LENGTH_KERNEL +
																									threadIdx.x];
		}
		__syncthreads();
		if ((threadIdx.x < (LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1)) && (threadIdx.y < (LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1)) &&
			(threadH < (inputH + LENGTH_KERNEL - 1)) && (threadW < (inputW + LENGTH_KERNEL - 1)))
		{
			for (int p = 0; p < LENGTH_KERNEL; p++)
			{
				for (int q = 0; q < LENGTH_KERNEL; q++)
				{
					if (((threadH + p) < (inputH + LENGTH_KERNEL - 1)) && ((threadW + q) < (inputW + LENGTH_KERNEL - 1)))
						acc += inTile[threadIdx.y + p][threadIdx.x + q] * weightFilter[p][q];
				}
			}
		}
		__syncthreads();
	}

	if ((threadIdx.x < (LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1)) && (threadIdx.y < (LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1)) &&
		(threadH < (inputH + LENGTH_KERNEL - 1)) && (threadW < (inputW + LENGTH_KERNEL - 1)))
	{
		output[outFeature * (inputH + LENGTH_KERNEL - 1) * (inputW + LENGTH_KERNEL - 1) + threadH * (inputW + LENGTH_KERNEL - 1) + threadW] = acc;
	}
}

__global__ void BackwardRelugrad(const double const* input, double* error, const int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size)
	{
		error[i] *= input[i] > 0;
	}
}

__global__ void BiasUpdate(double* bias, const double const* error, const int features, const int size)
{
	double acc = 0;
	if (threadIdx.x < features)
	{
		for (int i = 0; i < size; i++)
		{
			acc += error[threadIdx.x * size + i];
		}
		bias[threadIdx.x] = acc;
	}
}

__global__ void WeightConvoluteKernel(const double const* input, double* weight, const double const* output)
{
	double acc = 0;

	extern __shared__ double data[];

	data[threadIdx.y * blockDim.x + threadIdx.x] = input[blockIdx.x * blockDim.y * blockDim.x +
														 threadIdx.y * blockDim.x +
														 threadIdx.x];
	if ((threadIdx.y < (blockDim.y - LENGTH_KERNEL + 1)) && (threadIdx.x < (blockDim.x - LENGTH_KERNEL + 1)))
	{
		data[blockDim.y * blockDim.x + threadIdx.y * (blockDim.x - LENGTH_KERNEL + 1) + threadIdx.x] = output[blockIdx.y * (blockDim.y - LENGTH_KERNEL + 1) * (blockDim.x - LENGTH_KERNEL + 1) +
																											  threadIdx.y * (blockDim.x - LENGTH_KERNEL + 1) +
																											  threadIdx.x];
	}
	__syncthreads();

	if ((threadIdx.y < LENGTH_KERNEL) && (threadIdx.x < LENGTH_KERNEL))
	{
		for (int p = 0; p < blockDim.y - LENGTH_KERNEL + 1; p++)
		{
			for (int q = 0; q < blockDim.x - LENGTH_KERNEL + 1; q++)
			{
				acc += data[(threadIdx.y + p) * blockDim.x + (threadIdx.x + q)] * data[blockDim.y * blockDim.x + p * (blockDim.x - LENGTH_KERNEL + 1) + q];
			}
		}
		weight[blockIdx.x * gridDim.y * LENGTH_KERNEL * LENGTH_KERNEL +
			   blockIdx.y * LENGTH_KERNEL * LENGTH_KERNEL +
		       threadIdx.y * LENGTH_KERNEL +
			   threadIdx.x] = acc;
	}
}

void ConvolutionBackward(double* input, double* inError, double* outError, double* weight, double* weightDeltas, double* biasDeltas,
	const int inputFeatures, const int outputFeatures, const int inputWidth, const int inputHeight)
{
	/*Blocks are fixed sized tiles to allow for any size of input*/
	dim3 block(LENGTH_KERNEL_TILE, LENGTH_KERNEL_TILE, 1);
	unsigned int tilesW = ceil((float)(inputWidth) / (float)(LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1));
	unsigned int tilesH = ceil((float)(inputHeight) / (float)(LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1));
	dim3 grid(tilesW, tilesH, inputFeatures);
	ReverseConvoluteKernel <<< grid, block >>> (outError, inError, weight, outputFeatures, inputWidth - LENGTH_KERNEL + 1, inputHeight - LENGTH_KERNEL + 1);

	BackwardRelugrad <<< ceil(((float)(inputFeatures * inputWidth * inputHeight)) / ((float)(LENGTH_KERNEL_TILE * LENGTH_KERNEL_TILE))),
						LENGTH_KERNEL_TILE* LENGTH_KERNEL_TILE >>> (input, inError, inputFeatures * inputWidth * inputHeight);
	BiasUpdate <<< 1, outputFeatures >>> (biasDeltas, outError, outputFeatures, (inputWidth - LENGTH_KERNEL + 1) * (inputHeight - LENGTH_KERNEL + 1));
	
	/*Kernel does not support any size feature. ran out of time to make generalized*/
	block = dim3(inputWidth, inputHeight, 1);
	grid = dim3(inputFeatures, outputFeatures, 1);
	WeightConvoluteKernel <<< grid, block, (inputWidth * inputHeight + (inputWidth - LENGTH_KERNEL + 1) * (inputHeight - LENGTH_KERNEL + 1)) * sizeof(double) >>> 
								(input, weightDeltas, outError);
}

// Similar functionality as the code in Figure 16.5 of the textbook
#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	FOREACH(i, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH(**(output)))															\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
	}																							\
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)											\
{																								\
	const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));							\
	const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));							\
	FOREACH(i, GETLENGTH(outerror))																\
	FOREACH(o0, GETLENGTH(*(outerror)))															\
	FOREACH(o1, GETLENGTH(**(outerror)))														\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		inerror[i][o0*len0 + x0][o1*len1 + x1] = outerror[i][o0][o1];							\
	}																							\
}

#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)				\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			((double *)output)[y] += ((double *)input)[x] * weight[x][y];	\
	FOREACH(j, GETLENGTH(bias))												\
		((double *)output)[j] = action(((double *)output)[j] + bias[j]);	\
}

__global__ void CUDA_DotFInit(double* output, const double const* input, const double const* weight, const size_t w1size, const size_t w2size)
{
	const uint32_t ioutput = threadIdx.x + blockIdx.x * blockDim.x;

	if (ioutput < w2size)
	{
		double acc = 0.0;
		for (uint32_t x = 0; x < w1size; ++x)
			acc += input[x] * weight[ioutput + x * w2size];
		output[ioutput] = acc;
	}
}
__global__ void CUDA_DotFFinal(double* output, const double const* bias, const size_t w2size)
{
	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < w2size)
	{
		const double relux = output[tid] + bias[tid];
		output[tid] = relux * (relux > 0);
	}
}
void DotProductForward(double* input, double* output, double* weight, size_t w1size, size_t w2size, double* bias)
{
	// 1D
	{
		int32_t threads = w2size;
		int32_t blocks = 1;

		CUDA_DotFInit << <blocks, threads >> > (output, input, weight, w1size, w2size);
	}
	// 1D
	{
		int32_t threads = w2size;
		int32_t blocks = 1;
		CUDA_DotFFinal << <blocks, threads >> > (output, bias, w2size);
	}
}

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];printf("\ninerror\n");	\
	FOREACH(i, GETCOUNT(inerror))												\
		{((double *)inerror)[i] *= actiongrad(((double *)input)[i]); printf("%f ", inerror[i]);}printf("\nbd\n");				\
	FOREACH(j, GETLENGTH(outerror))												\
		{bd[j] += ((double *)outerror)[j]; printf("%f ", bd[j]);}printf("\nwd\n");										\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			{wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];	printf("%f ", wd[x][y]);}		\
}

__global__ void CUDA_DotBinerror(const double const* input, double* inerror, const double const* outerror, const double const* weight, double* bd, const size_t w1size, const size_t w2size)
{
	// 120x threads
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x < w1size)
	{
		double acc = 0.0;
		for (uint32_t y = 0; y < w2size; ++y)
		{
			acc += outerror[y] * weight[y + x * w2size];
		}
		inerror[x] = acc * input[x] > 0;
	}
	if (x < w2size)
		bd[x] += outerror[x];
}
__global__ void CUDA_DotBias(const double const* input, const double const* outerror, double* wd, const size_t w1size, const size_t w2size)
{
	// 8 blocks, 16 x 16 threads
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < w1size && y < w2size)
		wd[y + x * w2size] += input[x] * outerror[y];
}
void DotProductBackward(double* input, double* inerror, double* outerror, double* weight, double* wd, double* bd, const size_t w1size, const size_t w2size)
{
	{
		uint32_t threads = w1size;
		uint32_t blocks = 1;
		CUDA_DotBinerror<<<blocks, threads>>>(input, inerror, outerror, weight, bd, w1size, w2size);
	}
	{
		// 120 x 10
		dim3 threads = { 16, 16, 1 };
		dim3 blocks = {
			(uint32_t)ceil(((double)w1size / threads.x)),
			(uint32_t)ceil(((double)w2size / threads.y)),
			1 };
		CUDA_DotBias<<<blocks, threads>>>(input, outerror, wd, w1size, w2size);
	}
}



double relu(double x)
{
	return x*(x > 0);
}

double relugrad(double y)
{
	return y > 0;
}

static void forward(LeNet5 *lenet, Feature *features, double(*action)(double), LeNet5Cuda* lenetCuda, FeatureCuda* featuresCuda)
{
	CUDAMEMCPY_CHECK(lenet->weight0_1, lenetCuda->weight0_1, sizeof(lenet->weight0_1), cudaMemcpyHostToDevice);
	CUDAMEMCPY_CHECK(lenet->weight2_3, lenetCuda->weight2_3, sizeof(lenet->weight2_3), cudaMemcpyHostToDevice);
	CUDAMEMCPY_CHECK(lenet->weight4_5, lenetCuda->weight4_5, sizeof(lenet->weight4_5), cudaMemcpyHostToDevice);
	CUDAMEMCPY_CHECK(lenet->weight5_6, lenetCuda->weight5_6, sizeof(lenet->weight5_6), cudaMemcpyHostToDevice);
	CUDAMEMCPY_CHECK(lenet->bias0_1, lenetCuda->bias0_1, sizeof(lenet->bias0_1), cudaMemcpyHostToDevice);
	CUDAMEMCPY_CHECK(lenet->bias2_3, lenetCuda->bias2_3, sizeof(lenet->bias2_3), cudaMemcpyHostToDevice);
	CUDAMEMCPY_CHECK(lenet->bias4_5, lenetCuda->bias4_5, sizeof(lenet->bias4_5), cudaMemcpyHostToDevice);
	CUDAMEMCPY_CHECK(lenet->bias5_6, lenetCuda->bias5_6, sizeof(lenet->bias5_6), cudaMemcpyHostToDevice);

	ConvolutionForward(featuresCuda->input, featuresCuda->layer1, lenetCuda->weight0_1, lenetCuda->bias0_1,
					   INPUT, LAYER1, LENGTH_FEATURE0, LENGTH_FEATURE0);
	
	CUDAMEMCPY_CHECK(featuresCuda->layer1, features->layer1, sizeof(features->layer1), cudaMemcpyDeviceToHost);
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
	
	CUDAMEMCPY_CHECK(features->layer2, featuresCuda->layer2, sizeof(features->layer2), cudaMemcpyHostToDevice);
	ConvolutionForward(featuresCuda->layer2, featuresCuda->layer3, lenetCuda->weight2_3, lenetCuda->bias2_3,
					LAYER2, LAYER3, LENGTH_FEATURE2, LENGTH_FEATURE2);
	
	CUDAMEMCPY_CHECK(featuresCuda->layer3, features->layer3, sizeof(features->layer3), cudaMemcpyDeviceToHost);
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	
	CUDAMEMCPY_CHECK(features->layer4, featuresCuda->layer4, sizeof(features->layer4), cudaMemcpyHostToDevice);
	ConvolutionForward(featuresCuda->layer4, featuresCuda->layer5, lenetCuda->weight4_5, lenetCuda->bias4_5,
						LAYER4, LAYER5, LENGTH_FEATURE4, LENGTH_FEATURE4);

	DotProductForward(featuresCuda->layer5, featuresCuda->output, lenetCuda->weight5_6, LAYER5, OUTPUT, lenetCuda->bias5_6);
	CUDAMEMCPY_CHECK(featuresCuda->output, features->output, sizeof(features->output), cudaMemcpyDeviceToHost);
}

static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double), LeNet5Cuda* lenetCuda, LeNet5Cuda* deltasCuda, FeatureCuda* featuresCuda, FeatureCuda* errorsCuda)
{
	CUDAMEMCPY_CHECK(lenet->weight5_6, lenetCuda->weight5_6, sizeof(lenet->weight5_6), cudaMemcpyHostToDevice);
	CUDAMEMCPY_CHECK(features->layer5, featuresCuda->layer5, sizeof(features->layer5), cudaMemcpyHostToDevice);
	CUDAMEMCPY_CHECK(errors->output, errorsCuda->output, sizeof(errors->output), cudaMemcpyHostToDevice);
	DotProductBackward(featuresCuda->layer5, errorsCuda->layer5, errorsCuda->output, lenetCuda->weight5_6, deltasCuda->weight5_6, deltasCuda->bias5_6, LAYER5, OUTPUT);

	CUDAMEMCPY_CHECK(lenet->weight4_5, lenetCuda->weight4_5, sizeof(lenet->weight4_5), cudaMemcpyHostToDevice);
	CUDAMEMCPY_CHECK(features->layer4, featuresCuda->layer4, sizeof(features->layer4), cudaMemcpyHostToDevice);
	ConvolutionBackward(featuresCuda->layer4, errorsCuda->layer4, errorsCuda->layer5, lenetCuda->weight4_5, deltasCuda->weight4_5, deltasCuda->bias4_5,
						LAYER4, LAYER5, LENGTH_FEATURE4, LENGTH_FEATURE4);
	CUDAMEMCPY_CHECK(deltasCuda->weight4_5, deltas->weight4_5, sizeof(deltas->weight4_5), cudaMemcpyDeviceToHost);
	CUDAMEMCPY_CHECK(deltasCuda->bias4_5, deltas->bias4_5, sizeof(deltas->bias4_5), cudaMemcpyDeviceToHost);

	CUDAMEMCPY_CHECK(errorsCuda->layer4, errors->layer4, sizeof(errors->layer4), cudaMemcpyDeviceToHost);
	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	
	CUDAMEMCPY_CHECK(lenet->weight2_3, lenetCuda->weight2_3, sizeof(lenet->weight2_3), cudaMemcpyHostToDevice);
	CUDAMEMCPY_CHECK(features->layer2, featuresCuda->layer2, sizeof(features->layer2), cudaMemcpyHostToDevice);
	CUDAMEMCPY_CHECK(errors->layer3, errorsCuda->layer3, sizeof(errors->layer3), cudaMemcpyHostToDevice);
	ConvolutionBackward(featuresCuda->layer2, errorsCuda->layer2, errorsCuda->layer3, lenetCuda->weight2_3, deltasCuda->weight2_3, deltasCuda->bias2_3,
					LAYER2, LAYER3, LENGTH_FEATURE2, LENGTH_FEATURE2);
	CUDAMEMCPY_CHECK(deltasCuda->weight2_3, deltas->weight2_3, sizeof(deltas->weight2_3), cudaMemcpyDeviceToHost);
	CUDAMEMCPY_CHECK(deltasCuda->bias2_3, deltas->bias2_3, sizeof(deltas->bias2_3), cudaMemcpyDeviceToHost);

	CUDAMEMCPY_CHECK(errorsCuda->layer2, errors->layer2, sizeof(errors->layer2), cudaMemcpyDeviceToHost);
	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	
	CUDAMEMCPY_CHECK(lenet->weight0_1, lenetCuda->weight0_1, sizeof(lenet->weight0_1), cudaMemcpyHostToDevice);
	CUDAMEMCPY_CHECK(features->input, featuresCuda->input, sizeof(features->input), cudaMemcpyHostToDevice);
	CUDAMEMCPY_CHECK(errors->layer1, errorsCuda->layer1, sizeof(errors->layer1), cudaMemcpyHostToDevice);
	ConvolutionBackward(featuresCuda->input, errorsCuda->input, errorsCuda->layer1, lenetCuda->weight0_1, deltasCuda->weight0_1, deltasCuda->bias0_1,
					INPUT, LAYER1, LENGTH_FEATURE0, LENGTH_FEATURE0);
	CUDAMEMCPY_CHECK(deltasCuda->weight0_1, deltas->weight0_1, sizeof(deltas->weight0_1), cudaMemcpyDeviceToHost);
	CUDAMEMCPY_CHECK(deltasCuda->bias0_1, deltas->bias0_1, sizeof(deltas->bias0_1), cudaMemcpyDeviceToHost);

	CUDAMEMCPY_CHECK(errorsCuda->input, errors->input, sizeof(errors->input), cudaMemcpyDeviceToHost);
}

static inline void load_input(FeatureCuda *features, image input)
{
	double layer0[LENGTH_FEATURE0][LENGTH_FEATURE0];
	const long sz = sizeof(image) / sizeof(**input);
	double mean = 0, std = 0;
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		mean += input[j][k];
		std += input[j][k] * input[j][k];
	}
	mean /= sz;
	std = sqrt(std / sz - mean*mean);
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		layer0[j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
	}
	CUDAMEMCPY_CHECK(layer0, features->input, LENGTH_FEATURE0 * LENGTH_FEATURE0 * sizeof(double), cudaMemcpyHostToDevice);
}

static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)
{
	double inner = 0;
	for (int i = 0; i < count; ++i)
	{
		double res = 0;
		for (int j = 0; j < count; ++j)
		{
			res += exp(input[j] - input[i]);
		}
		loss[i] = 1. / res;
		inner -= loss[i] * loss[i];
	}
	inner += loss[label];
	for (int i = 0; i < count; ++i)
	{
		loss[i] *= (i == label) - loss[i] - inner;
	}
}

static void load_target(Feature *features, Feature *errors, int label)
{
	double *output = (double *)features->output;
	double *error = (double *)errors->output;
	softmax(output, error, label, GETCOUNT(features->output));
}

static uint8 get_result(Feature *features, uint8 count)
{
	double *output = (double *)features->output; 
	const int outlen = GETCOUNT(features->output);
	uint8 result = 0;
	double maxvalue = *output;
	for (uint8 i = 1; i < count; ++i)
	{
		if (output[i] > maxvalue)
		{
			maxvalue = output[i];
			result = i;
		}
	}
	return result;
}

static double f64rand()
{
	static int randbit = 0;
	if (!randbit)
	{
		//srand((unsigned)time(0));
		srand(0);
		for (int i = RAND_MAX; i; i >>= 1, ++randbit);
	}
	unsigned long long lvalue = 0x4000000000000000L;
	int i = 52 - randbit;
	for (; i > 0; i -= randbit)
		lvalue |= (unsigned long long)rand() << i;
	lvalue |= (unsigned long long)rand() >> -i;
	return *(double *)&lvalue - 3;
}

void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize, LeNet5Cuda* lenetCuda, LeNet5Cuda* deltasCuda, FeatureCuda* featuresCuda, FeatureCuda* errorsCuda)
{
	double buffer[GETCOUNT(LeNet5)] = { 0 };
	int i = 0;
	for (i = 0; i < batchSize; ++i)
	{ // For each training image
		// should be able to delete these once all parts are moved to cuda
		Feature features = { 0 };
		Feature errors = { 0 };
		LeNet5	deltas = { 0 };

		load_input(featuresCuda, inputs[i]);
		forward(lenet, &features, relu, lenetCuda, featuresCuda); // Forward propagation
		load_target(&features, &errors, labels[i]);

		backward(lenet, &deltas, &errors, &features, relugrad, lenetCuda, deltasCuda, featuresCuda, errorsCuda); // Backpropagation
		FOREACH(j, GETCOUNT(LeNet5))
				buffer[j] += ((double *)&deltas)[j];
	}
	double k = ALPHA / batchSize;
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += k * buffer[i];
}

/*
void Train(LeNet5 *lenet, image input, uint8 label)
{
	Feature features = { 0 };
	Feature errors = { 0 };
	LeNet5 deltas = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	load_target(&features, &errors, label);
	backward(lenet, &deltas, &errors, &features, relugrad);
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += ALPHA * ((double *)&deltas)[i];
}
*/

uint8 Predict(LeNet5 *lenet, image input,uint8 count, LeNet5Cuda* lenetCuda, FeatureCuda* featuresCuda)
{
	Feature features = { 0 };
	load_input(featuresCuda, input);
	forward(lenet, &features, relu, lenetCuda, featuresCuda);
	return get_result(&features, count);
}

void Initial(LeNet5 *lenet)
{
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->bias0_1; *pos++ = f64rand());
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
	for (double *pos = (double *)lenet->weight2_3; pos < (double *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
	for (double *pos = (double *)lenet->weight4_5; pos < (double *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
	for (double *pos = (double *)lenet->weight5_6; pos < (double *)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
	for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0);
}

void PrintResult(int confusion_matrix[OUTPUT][OUTPUT])
{
	// Print the confusion matrix
	printf("%15sPredicted label\n%10s", " ", " ");
	for (int col = 0; col < 10; col++)
		printf("%6d", col);
	printf("%10s\n", "Total");
	for (int n = 0; n < 70; n++)
		printf("%s", "-");
	printf("\nTrue label\n");
	int row_labels = 0;
	int total = 0;
	for (int row = 0; row < 10; row++) {
		row_labels = 0;
		printf("%10d", row);
		for (int col = 0; col < 10; col++) {
			printf("%6d", confusion_matrix[row][col]);
			row_labels += confusion_matrix[row][col];
		}
		printf("%10d\n", row_labels);
		total += row_labels;
	}
	for (int n = 0; n < 70; n++)
		printf("%s", "-");
	printf("\n%67s = %10d\n", "Total number of input images tested", total);
	for (int n = 0; n < 70; n++)
		printf("%s", "-");
	printf("\n");
}
