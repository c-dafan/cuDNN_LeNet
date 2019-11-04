#pragma once
#include"util.h"
#include<opencv2\opencv.hpp>
#include<opencv2\core.hpp>
#include<opencv2\highgui.hpp>

class CPUTensor
{
public:
	CPUTensor(int w, int h, int channels, int batch_size, float* data);
	CPUTensor(int w, int h, int channels, int batch_size, vector<cv::Mat> inputs);
	~CPUTensor();
	int batch_size, channels, height, width;
	float* data;
};


class CPUConv2D
{
public:
	CPUConv2D(int in_channel = 3, int out_channels = 3, int kernel_height = 3, int kernel_width = 3, int pad = 1, int stride = 1);
	~CPUConv2D();
	CPUTensor operator()(CPUTensor &input, Conv_W conv_w);

private:
	int in_channel, out_channels, kernel_height, kernel_width, pad, stride;
};



class CPUDense
{
public:
	CPUDense(int in_channel = 3, int out_channels = 3);
	~CPUDense();
	CPUTensor operator()(CPUTensor& input, Dense_W weight);
private:

	int in_channels;
	int out_channels;
};


class CPUActivateLayer
{
public:
	CPUActivateLayer();
	~CPUActivateLayer();
	CPUTensor operator()(CPUTensor& inputs);
private:

};

class CPUPoolingLayer
{
public:
	CPUPoolingLayer(int size = 2);
	~CPUPoolingLayer();
	CPUTensor operator()(CPUTensor& inputs);

private:
	int size;
};

