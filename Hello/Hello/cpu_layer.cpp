#include "cpu_layer.h"

CPUConv2D::CPUConv2D(int in_channel, int out_channels, int kernel_height, int kernel_width, int pad, int stride) :in_channel(in_channel),
	out_channels(out_channels), kernel_height(kernel_height), kernel_width(kernel_width), pad(pad), stride(stride)
{

}

CPUConv2D::~CPUConv2D()
{
}


CPUTensor CPUConv2D::operator()(CPUTensor & input, Conv_W conv_w)
{
	int out_width = (input.width + 2 * pad - kernel_width) / stride + 1;
	int out_height = (input.height + 2 * pad - kernel_height) / stride + 1;
	float *out_data = new float[out_channels * out_width * out_height * input.batch_size]();
	for (int bs = 0; bs < input.batch_size; bs++) {
		for (int ow = 0; ow < out_width; ow++) {
			for (int oh = 0; oh < out_height; oh++) {
				for (int co = 0; co < out_channels; co++) {
					for (int ci = 0; ci < in_channel; ci++) {
						for (int kw = 0; kw < kernel_width; kw++) {
							for (int kh = 0; kh < kernel_height; kh++) {
								out_data[bs*out_width*out_height*out_channels + ow*out_height*out_channels + oh*out_channels + co] +=
								input.data[bs*input.height*in_channel*input.width +(ow + kw) * input.height*in_channel +
								(oh + kh)* in_channel + ci]
								* conv_w.data[co][ci][kw][kh];
							}
						}
					}
				}
			}
		}
	}
	return CPUTensor(out_width, out_height, out_channels, input.batch_size, out_data);
}

CPUDense::CPUDense(int in_channel, int out_channels):in_channels(in_channel), out_channels(out_channels)
{

}

CPUDense::~CPUDense()
{
}

CPUTensor CPUDense::operator()(CPUTensor & input, Dense_W weight)
{
	float *out_data = new float[input.batch_size * out_channels];
	memset(out_data, 0, input.batch_size * out_channels * sizeof(float));
	for (int b = 0; b < input.batch_size; ++b) {
		for (int out = 0; out < out_channels; out++) {
			for (int in = 0; in < in_channels; in++) {
				out_data[b * out_channels + out] += input.data[b * in_channels + in] * weight.data[out][in];
			}
		}
	}
	return CPUTensor(1,1, out_channels, input.batch_size,out_data);
}

CPUActivateLayer::CPUActivateLayer()
{
}

CPUActivateLayer::~CPUActivateLayer()
{
}

CPUTensor CPUActivateLayer::operator()(CPUTensor& inputs)
{
	float* out_data = new float[inputs.batch_size * inputs.channels * inputs.height * inputs.width]();
	for (int i = 0; i < inputs.batch_size * inputs.channels * inputs.height * inputs.width; i++) {
		*(out_data + i) = max((float)0, *(inputs.data + i));
	}

	return CPUTensor(inputs.width, inputs.height, inputs.channels,inputs.batch_size, out_data);
}

CPUPoolingLayer::CPUPoolingLayer(int size):size(size)
{
}

CPUPoolingLayer::~CPUPoolingLayer()
{
}

CPUTensor CPUPoolingLayer::operator()(CPUTensor & inputs)
{
	int out_width = inputs.width / size;
	int out_height = inputs.height / size;
	float *data = new float[out_width * out_height *inputs.channels * inputs.batch_size];
	for (int bs = 0; bs < inputs.batch_size; bs++) {
		for (int ch = 0; ch < inputs.channels; ch++) {
			for (int ow = 0; ow < out_width; ow++) {
				for (int oh = 0; oh < out_height; oh++) {
					data[ch + ow*out_height*inputs.channels + oh*inputs.channels + bs*inputs.channels*out_width * out_height] =
						inputs.data[ch + (ow*size)*inputs.height*inputs.channels + (oh*size)*inputs.channels + bs*inputs.channels*inputs.width  * inputs.height];
					for (int ws = 0; ws < size; ws++) {
						for (int wh = 0; wh < size; wh++) {
							data[ch + ow*out_height*inputs.channels + oh*inputs.channels + bs*inputs.channels*out_width * out_height] =
								max(data[ch + ow*out_height*inputs.channels + oh*inputs.channels + bs*inputs.channels*out_width * out_height],
									inputs.data[ch + (ow*size + ws)*inputs.height*inputs.channels + (oh*size + wh)*inputs.channels
									+bs*inputs.channels*inputs.width  * inputs.height]);
						}
					}
				}
			}
		}
	}

	return CPUTensor(out_width, out_height, inputs.channels, inputs.batch_size, data);
}

CPUTensor::CPUTensor(int w, int h, int channels, int batch_size, float * data):width(w), height(h), channels(channels), batch_size(batch_size), data(data)
{

}
CPUTensor::CPUTensor(int w, int h, int channels, int batch_size, vector<cv::Mat> inputs) : width(w), height(h), channels(channels), batch_size(batch_size) {
	int src_image_bytes = w * h*channels*batch_size;
	float *data = new float[src_image_bytes];
	float* data_tmp = data;
	for (auto img : inputs) {
		for (int i = 0; i < img.cols * img.rows * img.channels(); i++) {
			*data_tmp = img.at<float>(i);
			data_tmp++;
		}
	}
	this->data = data;
}

CPUTensor::~CPUTensor()
{
	if (data != nullptr) {
		delete[] data;
		data = nullptr;
	}
}
