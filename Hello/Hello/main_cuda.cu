#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <iostream>
using namespace std;

#define arraySize 4

__global__ void MatMul(float *C, float *A, float *B, int width, int b_width, int data_len)
{
	unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id > data_len) {
		unsigned int w = id / b_width;
		unsigned int h = id % b_width;
		float sum = 0.0F;
		for (int j = 0; j < width; j++) {
			sum += A[w * width + j] * B[j * b_width + h];
		}
		*(C + id) = sum;
	}
}

__global__ void ReLU(float *C, float *A, int data_len)
{
	unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < data_len) {
		if (A[id] > 0) {
			C[id] = A[id];
		}
		else {
			C[id] = 0;
		}
	}
}

__device__ float maxx(float a, float b) {
	if (a > b) {
		return a;
	}
	return b;
}


__global__ void MaxPooling(float *data, float *inputs, int size, int out_height, int out_width, int in_width, int in_height, int in_channels)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	int bs = id / (in_channels * out_height * out_width);
	int ow = (id - bs * in_channels * out_height * out_width) / (in_channels * out_height);
	int oh = (id - bs * in_channels * out_height * out_width - ow * in_channels * out_height)/ in_channels;
	int ch = id - bs * in_channels * out_height * out_width - ow * in_channels * out_height - oh * in_channels;
	float mmax = inputs[ch + ow*size*in_height*in_channels + oh*size*in_channels + bs*in_channels*in_width  * in_height];
	for (int ws = 0; ws < size; ws++) {
		for (int wh = 0; wh < size; wh++) {
			mmax = maxx(mmax, inputs[ch + (ow*size + ws)*in_height*in_channels + (oh*size + wh)*in_channels
					+ bs*in_channels*in_width  * in_height]);
		}
	}
	data[id] = mmax;
}

__global__ void CONV(float *out_data, float *input,float *weight, int kernel_size,int out_width,int out_height,int out_channels, 
	int in_width, int in_height, int in_channels)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int bs = id / (out_channels * out_height * out_width);
	int ow = (id - bs * out_channels * out_height * out_width) / (out_channels * out_height);
	int oh = (id - bs * out_channels * out_height * out_width - ow * out_channels * out_height) / out_channels;
	int co = id - bs * out_channels * out_height * out_width - ow * out_channels * out_height -oh * out_channels;
	float result = 0;
	for (int ci = 0; ci < in_channels; ci++) {
		for (int kw = 0; kw < kernel_size; kw++) {
			for (int kh = 0; kh < kernel_size; kh++) {
				result +=
					input[bs*in_height*in_channels*in_width + (ow * kernel_size + kw) * in_height*in_channels +
					(oh * kernel_size + kh)* in_channels + ci]
					* weight[co * in_channels*kernel_size*kernel_size + ci *kernel_size*kernel_size + kw * kernel_size + kh];
			}
		}
	}
	out_data[id] = result;
}

int main()
{

	float A[arraySize][arraySize] = { { 1, 2, 3 , 4 } , { 1, 2, 3 , 4},{ 4, 5 ,6 ,7},{ 7, 8, 9, 10 } };
	float weight[arraySize / 2][arraySize / 2] = { {1, 1}, {1, 1} };
	//float B[arraySize][arraySize] = { { 1, 0, 0 },{ 0, 1 ,0 },{ 0, 0, 1 } };
	//float C[arraySize][arraySize] = { { 1, 1, 1 },{ 1, 1 , 1 },{ 1, 1, 1 } };
	float D[arraySize/2][arraySize/2] = { { 0,0 },{ 0,0 }};
	float *dev_A;
	float *dev_D;
	float *dev_weight;
	/*float *dev_B;
	float *dev_C;*/

	cudaMalloc(&dev_A, arraySize*arraySize * sizeof(float));
	cudaMalloc(&dev_D, arraySize*arraySize / 4 * sizeof(float));
	cudaMalloc(&dev_weight, arraySize*arraySize/4 * sizeof(float));

	//cudaMalloc(&dev_B, arraySize*arraySize * sizeof(float));
	//cudaMalloc(&dev_C, arraySize*arraySize * sizeof(float));

	cudaMemcpy(dev_A, A, arraySize*arraySize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_weight, weight, arraySize*arraySize/4 * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_B, B, arraySize*arraySize * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_C, C, arraySize*arraySize*sizeof(float), cudaMemcpyHostToDevice);


	//MatMul << <1, arraySize*arraySize >> >(dev_C, dev_A, dev_B, 3, 3);
	//ReLU << <1, arraySize*arraySize >> > (dev_C, dev_A, arraySize*arraySize);
	//MaxPooling<<<1, 4>>>(dev_D, dev_A, 2, 2, 2, 4, 4, 1);
	CONV<<<1, 4>>>(dev_D, dev_A, dev_weight, 2, 2, 2, 1, 4, 4, 1);
	cudaMemcpy(D, dev_D, arraySize*arraySize/4 * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < arraySize/2; i++)
	{
		for (int j = 0; j < arraySize/2; j++)
		{
			printf("C[%d][%d] = %f \t", i, j, D[i][j]);
		}
		printf("\n");
	}

	cudaFree(dev_D);
	cudaFree(dev_A);
	//cudaFree(dev_B);

}
