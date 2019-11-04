#pragma once
#include <cudnn.h>
#include"util.h"
#include<opencv2\opencv.hpp>
#include<opencv2\core.hpp>
#include<opencv2\highgui.hpp>

void createTensor(cudnnTensorDescriptor_t& input_descriptor, int w, int h, int channels, int batch_size);
void createConv(cudnnConvolutionDescriptor_t& convolution_descriptor, int pad = 1, int stride = 1);
void createKernel(cudnnFilterDescriptor_t& kernel_descriptor, int in_channel = 3, int out_channels = 3, int kernel_height = 3, int kernel_width = 3);

class Tensor
{
public:
	Tensor(int w, int h, int channels, int batch_size, float* data, bool is_cuda);
	Tensor(int w, int h, int channels, int batch_size, vector<cv::Mat> inputs);
	~Tensor();
	float *get_data();
//private:
	cudnnTensorDescriptor_t input_descriptor;
	int batch_size, channels, height, width;
	float* data;
	float* gpu_data;

};


class Conv2D
{
public:
	Conv2D(cudnnHandle_t& cudnn, Conv_W conv_w, int in_channel = 3, int out_channels = 3, int kernel_height = 3, int kernel_width = 3, int pad = 1, int stride = 1);
	~Conv2D();
	Tensor operator()(Tensor &input/*, Conv_W conv_w*/);

private:
	cudnnConvolutionDescriptor_t convolution_descriptor;
	cudnnFilterDescriptor_t kernel_descriptor;
	cudnnHandle_t cudnn;
	float *d_kernel;
};



class Dense
{
public:
	Dense(cudnnHandle_t& cudnn, Dense_W weight, int in_channel = 3, int out_channels = 3);
	~Dense();
	Tensor operator()(Tensor& input/*, Dense_W weight*/);
private:
	cudnnConvolutionDescriptor_t convolution_descriptor;
	cudnnFilterDescriptor_t kernel_descriptor;
	cudnnHandle_t cudnn;
	int in_channels;
	int out_channels;
	float *d_kernel;
};


class ActivateLayer
{
public:
	ActivateLayer(cudnnHandle_t& cudnn);
	~ActivateLayer();
	Tensor operator()(Tensor& inputs);
private: 
	cudnnHandle_t cudnn;
	cudnnActivationDescriptor_t activation_descriptor;
};

class PoolingLayer
{
public:
	PoolingLayer(cudnnHandle_t& handel, int size=2);
	~PoolingLayer();
	Tensor operator()(Tensor& inputs);

private:
	cudnnPoolingDescriptor_t pooling_decriptor;
	cudnnHandle_t handel;
};

