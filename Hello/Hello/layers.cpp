#include"layers.h"
#include<iostream>
#include <numeric>
#include<vector>

void createTensor(cudnnTensorDescriptor_t& input_descriptor, int w, int h, int channels, int batch_size) {
	cudnnCreateTensorDescriptor(&input_descriptor);
	cudnnSetTensor4dDescriptor(input_descriptor,
		/*format=*/CUDNN_TENSOR_NHWC,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/batch_size,
		/*channels=*/channels,
		/*image_height=*/h,
		/*image_width=*/w);
}

void createConv(cudnnConvolutionDescriptor_t& convolution_descriptor, int pad, int stride) {
	// 卷积操作的描述（步长、填充等等）
	cudnnCreateConvolutionDescriptor(&convolution_descriptor);
	cudnnSetConvolution2dDescriptor(convolution_descriptor,
		/*pad_height=*/pad,
		/*pad_width=*/pad,
		/*vertical_stride=*/stride,
		/*horizontal_stride=*/stride,
		/*dilation_height=*/1,
		/*dilation_width=*/1,
		/*mode=*/CUDNN_CROSS_CORRELATION, // CUDNN_CONVOLUTION
		/*computeType=*/CUDNN_DATA_FLOAT);
}

void createKernel(cudnnFilterDescriptor_t& kernel_descriptor, int in_channel, int out_channels, int kernel_height, int kernel_width) {
	cudnnCreateFilterDescriptor(&kernel_descriptor);
	cudnnSetFilter4dDescriptor(kernel_descriptor,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*format=*/CUDNN_TENSOR_NCHW,	// 注意是 NCHW
		/*out_channels=*/out_channels,
		/*in_channels=*/in_channel,
		/*kernel_height=*/kernel_height,
		/*kernel_width=*/kernel_width);
}
float*  Tensor::get_data() {
	if (data == nullptr) {
		int src_image_bytes = batch_size * width * height * channels;
		float *da = new float[src_image_bytes];
		cudaMemcpy(da, gpu_data, src_image_bytes * sizeof(float), cudaMemcpyDeviceToHost);
		data = da;
	}
	return data;
}

Tensor::Tensor(int w, int h, int channels, int batch_size, vector<cv::Mat> inputs) : width(w), height(h), channels(channels), batch_size(batch_size) {
	int src_image_bytes = w * h * channels * batch_size;
	float *data = new float[src_image_bytes];
	float* data_tmp = data;
	for (auto img : inputs) {
		for (int i = 0; i < img.cols * img.rows * img.channels(); i++) {
			*data_tmp = img.at<float>(i);
			data_tmp++;
		}
	}
	this->data = data;
	cudaMalloc(&gpu_data, src_image_bytes * sizeof(float));
	cudaMemcpy(gpu_data, data, src_image_bytes * sizeof(float), cudaMemcpyHostToDevice);
	createTensor(input_descriptor, width, height, channels, batch_size);
}

Tensor::Tensor(int w, int h, int channels, int batch_size, float* data, bool is_cuda) : width(w), height(h), channels(channels), batch_size(batch_size)
{
	createTensor(input_descriptor, width, height, channels, batch_size);
	if (is_cuda) {
		gpu_data = data;
		this->data = nullptr;
	}
	else {
		this->data = data;
		int src_image_bytes = w * h * channels * batch_size;
		cudaMalloc(&gpu_data, src_image_bytes * sizeof(float));
		cudaMemcpy(gpu_data, data, src_image_bytes * sizeof(float), cudaMemcpyHostToDevice);
	}
}

Tensor::~Tensor()
{
	if (data != nullptr) {
		delete[] data;
		data = nullptr;
	}
	cudaFree(gpu_data);
	cudnnDestroyTensorDescriptor(input_descriptor);
}


Conv2D::Conv2D(cudnnHandle_t& cudnn, Conv_W conv_w, int in_channel, int out_channels, int kernel_height, int kernel_width, int pad, int stride) :cudnn(cudnn)
{
	float* h_kernel = new float[conv_w.n * conv_w.c * conv_w.w * conv_w.h];
	for (int n = 0; n < conv_w.n; n++) {
		for (int c = 0; c < conv_w.c; c++) {
			for (int w = 0; w < conv_w.w; w++) {
				for (int h = 0; h < conv_w.h; h++) {
					h_kernel[n*conv_w.c * conv_w.w * conv_w.h +
						c * conv_w.w * conv_w.h +
						w *  conv_w.h + h] = conv_w.data[n][c][w][h];
				}
			}
		}
	}
	cudaMalloc(&d_kernel, conv_w.n * conv_w.c * conv_w.w * conv_w.h * sizeof(float));
	cudaMemcpy(d_kernel, h_kernel, conv_w.n * conv_w.c * conv_w.w * conv_w.h * sizeof(float), cudaMemcpyHostToDevice);
	delete[] h_kernel;
	createKernel(kernel_descriptor, in_channel, out_channels, kernel_height, kernel_width);
	createConv(convolution_descriptor, pad, stride);
}

Conv2D::~Conv2D()
{
	cudnnDestroyFilterDescriptor(kernel_descriptor);
	cudnnDestroyConvolutionDescriptor(convolution_descriptor);
	cudaFree(d_kernel);
}


Tensor Conv2D::operator()(Tensor &input/*, Conv_W conv_w*/) {

	int batch_size{ 0 }, channels{ 0 }, height{ 0 }, width{ 0 };
	cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
		input.input_descriptor,
		kernel_descriptor,
		&batch_size,
		&channels,
		&height,
		&width);
	cudnnTensorDescriptor_t output_descriptor;
	cudnnCreateTensorDescriptor(&output_descriptor);
	cudnnSetTensor4dDescriptor(output_descriptor,
		/*format=*/CUDNN_TENSOR_NHWC,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/batch_size,
		/*channels=*/channels,
		/*image_height=*/height,
		/*image_width=*/width);

	cudnnConvolutionFwdAlgo_t convolution_algorithm;

	cudnnGetConvolutionForwardAlgorithm(cudnn,
		input.input_descriptor,
		kernel_descriptor,
		convolution_descriptor,
		output_descriptor,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		/*memoryLimitInBytes=*/0,
		&convolution_algorithm);

	size_t workspace_bytes{ 0 };
	cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		input.input_descriptor,
		kernel_descriptor,
		convolution_descriptor,
		output_descriptor,
		convolution_algorithm,
		&workspace_bytes);
	void* d_workspace{ nullptr };
	cudaMalloc(&d_workspace, workspace_bytes);

	float* d_output{ nullptr };
	int dst_image_bytes = batch_size *channels * height * width;;
	cudaMalloc(&d_output, dst_image_bytes * sizeof(float));

	const float alpha = 1.0f, beta = 0.0f;

	cudnnConvolutionForward(cudnn,
		&alpha,
		input.input_descriptor,
		input.gpu_data,
		kernel_descriptor,
		d_kernel,
		convolution_descriptor,
		convolution_algorithm,
		d_workspace, 
		workspace_bytes,
		&beta,
		output_descriptor,
		d_output);

	cudaFree(d_workspace);
	cudnnDestroyTensorDescriptor(output_descriptor);
	return Tensor(width, height, channels, batch_size, d_output, true);
}

Dense::Dense(cudnnHandle_t& cudnn,Dense_W weight, int in_channel, int out_channels) :cudnn(cudnn), in_channels(in_channel), out_channels(out_channels)
{
	createKernel(kernel_descriptor, in_channel, out_channels, 1, 1);
	createConv(convolution_descriptor, 0, 1);
	float *h_kernel = new float[in_channels * out_channels];

	for (int channel = 0; channel < out_channels; ++channel) {
		for (int kernel = 0; kernel < in_channels; ++kernel) {
			h_kernel[channel*in_channels + kernel] = weight.data[channel][kernel];
		}
	}

	cudaMalloc(&d_kernel, in_channels * out_channels * sizeof(float));
	cudaMemcpy(d_kernel, h_kernel, in_channels * out_channels * sizeof(float), cudaMemcpyHostToDevice);
	delete[] h_kernel;
}

Dense::~Dense()
{
	cudnnDestroyFilterDescriptor(kernel_descriptor);
	cudnnDestroyConvolutionDescriptor(convolution_descriptor);
	cudaFree(d_kernel);
}



Tensor Dense::operator()(Tensor &input/*, Dense_W weight*/) {
	int batch_size{ 0 }, channels{ 0 }, height{ 0 }, width{ 0 };
	cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
		input.input_descriptor,
		kernel_descriptor,
		&batch_size,
		&channels,
		&height,
		&width);
	cudnnTensorDescriptor_t output_descriptor;
	cudnnCreateTensorDescriptor(&output_descriptor);
	cudnnSetTensor4dDescriptor(output_descriptor,
		/*format=*/CUDNN_TENSOR_NHWC,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/batch_size,
		/*channels=*/channels,
		/*image_height=*/height,
		/*image_width=*/width);

	cudnnConvolutionFwdAlgo_t convolution_algorithm;

	cudnnGetConvolutionForwardAlgorithm(cudnn,
		input.input_descriptor,
		kernel_descriptor,
		convolution_descriptor,
		output_descriptor,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, // CUDNN_CONVOLUTION_FWD_SPECIFY_?WORKSPACE_LIMIT（在内存受限的情况下，memoryLimitInBytes 设置非 0 值）
		/*memoryLimitInBytes=*/0,
		&convolution_algorithm);
	size_t workspace_bytes{ 0 };
	cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		input.input_descriptor,
		kernel_descriptor,
		convolution_descriptor,
		output_descriptor,
		convolution_algorithm,
		&workspace_bytes);
	void* d_workspace{ nullptr };
	cudaMalloc(&d_workspace, workspace_bytes);


	int dst_image_bytes = batch_size * channels * height * width;

	float* d_output{ nullptr };
	cudaMalloc(&d_output, dst_image_bytes * sizeof(float));

	/*float *h_kernel = new float[in_channels * out_channels];


	for (int channel = 0; channel < out_channels; ++channel) {
		for (int kernel = 0; kernel < in_channels; ++kernel) {
			h_kernel[channel*in_channels +kernel] = weight.data[channel][kernel];
		}
	}

	float* d_kernel{ nullptr };
	cudaMalloc(&d_kernel, in_channels * out_channels * sizeof(float));
	cudaMemcpy(d_kernel, h_kernel, in_channels * out_channels * sizeof(float), cudaMemcpyHostToDevice);
	delete[] h_kernel;*/

	const float alpha = 1.0f, beta = 0.0f;

	cudnnConvolutionForward(cudnn,
		&alpha,
		input.input_descriptor,
		input.gpu_data,
		kernel_descriptor,
		d_kernel,
		convolution_descriptor,
		convolution_algorithm,
		d_workspace,  // 注意，如果我们选择不需要额外内存的卷积算法，d_workspace可以为nullptr。
		workspace_bytes,
		&beta,
		output_descriptor,
		d_output);

	cudaFree(d_kernel);
	cudaFree(d_workspace);
	cudnnDestroyTensorDescriptor(output_descriptor);
	return Tensor(width, height, channels, batch_size, d_output, true);
}

Tensor PoolingLayer::operator()(Tensor& inputs) {
	const float alpha = 1.0f, beta = 0.0f;
	int batch_size{ 0 }, channels{ 0 }, height{ 0 }, width{ 0 };
	cudnnGetPooling2dForwardOutputDim(pooling_decriptor, inputs.input_descriptor, &batch_size, &channels, &height, &width);

	cudnnTensorDescriptor_t output_descriptor;
	cudnnCreateTensorDescriptor(&output_descriptor);
	cudnnSetTensor4dDescriptor(output_descriptor,
		/*format=*/CUDNN_TENSOR_NHWC,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/batch_size,
		/*channels=*/channels,
		/*image_height=*/height,
		/*image_width=*/width);

	float* d_output{ nullptr };
	int dst_size = batch_size* channels*height*width;
	cudaMalloc(&d_output, dst_size * sizeof(float));
	cudnnPoolingForward(handel, pooling_decriptor, &alpha, inputs.input_descriptor,
		inputs.gpu_data, &beta, output_descriptor, d_output);
	
	cudnnDestroyTensorDescriptor(output_descriptor);
	return Tensor(width, height, channels, batch_size, d_output, true);
}


PoolingLayer::PoolingLayer(cudnnHandle_t& handel, int size) :handel(handel)
{
	cudnnCreatePoolingDescriptor(&pooling_decriptor);
	cudnnSetPooling2dDescriptor(pooling_decriptor, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, size, size, 0, 0, size, size);
}

PoolingLayer::~PoolingLayer()
{
	cudnnDestroyPoolingDescriptor(pooling_decriptor);
}

ActivateLayer::ActivateLayer(cudnnHandle_t& cudnn) :cudnn(cudnn)
{
	checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
	checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
		CUDNN_ACTIVATION_RELU,
		CUDNN_PROPAGATE_NAN,
		/*relu_coef=*/0));
}

ActivateLayer::~ActivateLayer()
{
	cudnnDestroyActivationDescriptor(activation_descriptor);
}

Tensor ActivateLayer::operator()(Tensor& inputs) {
	const float alpha = 1.0f, beta = 0.0f;

	int image_bytes = inputs.batch_size * inputs.channels * inputs.height * inputs.width;

	float* d_output{ nullptr };
	cudaMalloc(&d_output, image_bytes * sizeof(float));

	// 前向 sigmoid 激活函数
	cudnnActivationForward(cudnn,
		activation_descriptor,
		&alpha,
		inputs.input_descriptor,
		inputs.gpu_data,
		&beta,
		inputs.input_descriptor,
		d_output);

	return Tensor(inputs.width, inputs.height, inputs.channels, inputs.batch_size, d_output, true);
}

