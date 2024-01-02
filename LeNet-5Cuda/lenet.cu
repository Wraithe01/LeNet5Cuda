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

#define CUDAMEMCPY_CHECK(src, dest, bytes, type)												 \
	if (cudaMemcpy(dest, src, bytes, type) != cudaSuccess)								 \
		fprintf(stderr, "ERROR: cudaMemCpy %s from %s to %s failed!\n", #type, #src, #dest); \


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

__global__ void ConvoluteKernelValid(double* input, double* output, double* weight, const int inputFeatures, const int inputW, const int inputH)
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
			weightFilter[threadIdx.y][threadIdx.x] = weight[inFeature * gridDim.z * LENGTH_KERNEL * LENGTH_KERNEL +
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
	}

	if ((threadIdx.x < (LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1)) && (threadIdx.y < (LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1)) &&
		(threadH < (inputH - LENGTH_KERNEL + 1)) && (threadW < (inputW - LENGTH_KERNEL + 1)))
	{
		output[outFeature * (inputH - LENGTH_KERNEL + 1) * (inputW - LENGTH_KERNEL + 1) + threadH * (inputW - LENGTH_KERNEL + 1) + threadW] = acc;
	}
}

void ConvoluteValid(double* input, double* output, double* weight, const int inputFeatures, const int outputFeatures, const int inputWidth, const int inputHeight)
{
	/*Blocks are fixed sized tiles to allow for any size of input*/
	dim3 block(LENGTH_KERNEL_TILE, LENGTH_KERNEL_TILE, 1);
	unsigned int tilesW = ceil((float)(inputWidth - LENGTH_KERNEL + 1) / (float)(LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1));
	unsigned int tilesH = ceil((float)(inputHeight - LENGTH_KERNEL + 1) / (float)(LENGTH_KERNEL_TILE - LENGTH_KERNEL + 1));
	dim3 grid(tilesW, tilesH, outputFeatures);
	ConvoluteKernelValid <<< grid, block >>> (input, output, weight, inputFeatures, inputWidth, inputHeight);
}

__global__ void ForwardReluKernel(double* feature, double* bias, const int featureWidth, const int featureHeight)
{
	int width = blockIdx.x * LENGTH_KERNEL_TILE + threadIdx.x;
	int height = blockIdx.y * LENGTH_KERNEL_TILE + threadIdx.y;
	int featureMap = blockIdx.z;

	if ((width < featureWidth) && (height < featureHeight))
	{
		if ((feature[featureMap * featureHeight * featureWidth + height * featureWidth + width] + bias[featureMap]) < 0)
			feature[featureMap * featureHeight * featureWidth + height * featureWidth + width] = 0;
	}
}

void ForwardRelu(double* feature, double* bias, const int featureCount, const int featureWidth, const int featureHeight)
{
	/*Blocks are fixed sized tiles to allow for any size of input*/
	dim3 block(LENGTH_KERNEL_TILE, LENGTH_KERNEL_TILE, 1);
	unsigned int tilesW = ceil((float)featureWidth / (float)LENGTH_KERNEL_TILE);
	unsigned int tilesH = ceil((float)featureHeight / (float)LENGTH_KERNEL_TILE);
	dim3 grid(tilesW, tilesH, featureCount);
	ForwardReluKernel <<< grid, block >>> (feature, bias, featureWidth, featureHeight);
}

// Similar functionality as the code in Figure 16.4 of the textbook
void ConvolutionForward(double* input, double* output, double* weight, double* bias, const int inputFeatures, const int outputFeatures, const int inputWidth, const int inputHeight)					\
{
	ConvoluteValid(input, output, weight, inputFeatures, outputFeatures, inputWidth, inputHeight);
	ForwardRelu(output, bias, outputFeatures, inputWidth - LENGTH_KERNEL + 1, inputHeight - LENGTH_KERNEL + 1);
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);				\
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

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];	\
	FOREACH(i, GETCOUNT(inerror))												\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);				\
	FOREACH(j, GETLENGTH(outerror))												\
		bd[j] += ((double *)outerror)[j];										\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];			\
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

	CUDAMEMCPY_CHECK(features->input, featuresCuda->input, sizeof(features->input), cudaMemcpyHostToDevice);
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
	
	CUDAMEMCPY_CHECK(featuresCuda->layer5, features->layer5, sizeof(features->layer5), cudaMemcpyDeviceToHost);
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}

static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
	DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
	CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
}

static inline void load_input(Feature *features, image input)
{
	double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
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
		layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
	}
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

		load_input(&features, inputs[i]);
		forward(lenet, &features, relu, lenetCuda, featuresCuda); // Forward propagation
		load_target(&features, &errors, labels[i]);
		/*
		CUDAMEMCPY_CHECK(deltas.weight0_1, deltasCuda->weight0_1, cudaMemcpyHostToDevice);
		CUDAMEMCPY_CHECK(deltas.weight2_3, deltasCuda->weight2_3, cudaMemcpyHostToDevice);
		CUDAMEMCPY_CHECK(deltas.weight4_5, deltasCuda->weight4_5, cudaMemcpyHostToDevice);
		CUDAMEMCPY_CHECK(deltas.weight5_6, deltasCuda->weight5_6, cudaMemcpyHostToDevice);
		CUDAMEMCPY_CHECK(deltas.bias0_1, deltasCuda->bias0_1, cudaMemcpyHostToDevice);
		CUDAMEMCPY_CHECK(deltas.bias2_3, deltasCuda->bias2_3, cudaMemcpyHostToDevice);
		CUDAMEMCPY_CHECK(deltas.bias4_5, deltasCuda->bias4_5, cudaMemcpyHostToDevice);
		CUDAMEMCPY_CHECK(deltas.bias5_6, deltasCuda->bias5_6, cudaMemcpyHostToDevice);

		CUDAMEMCPY_CHECK(errors.input, errorsCuda->input, cudaMemcpyHostToDevice);
		CUDAMEMCPY_CHECK(errors.layer1, errorsCuda->layer1, cudaMemcpyHostToDevice);
		CUDAMEMCPY_CHECK(errors.layer2, errorsCuda->layer2, cudaMemcpyHostToDevice);
		CUDAMEMCPY_CHECK(errors.layer3, errorsCuda->layer3, cudaMemcpyHostToDevice);
		CUDAMEMCPY_CHECK(errors.layer4, errorsCuda->layer4, cudaMemcpyHostToDevice);
		CUDAMEMCPY_CHECK(errors.layer5, errorsCuda->layer5, cudaMemcpyHostToDevice);
		CUDAMEMCPY_CHECK(errors.output, errorsCuda->output, cudaMemcpyHostToDevice);
		*/

		backward(lenet, &deltas, &errors, &features, relugrad); // Backpropagation
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
	load_input(&features, input);
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
