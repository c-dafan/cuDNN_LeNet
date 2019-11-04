#include<iostream>
#include"util.h"
#include"cpu_layer.h"
#include<algorithm>
#include <cudnn.h>
#include"layers.h"
#include <numeric>
#include <chrono>

using namespace std;


void main(int argc, const char* argv[]) {
	if (argc < 2) {
		std::cerr << "usage: conv <image> [gpu=0] [sigmoid=0]" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	int gpu_id = (argc > 2) ? std::atoi(argv[2]) : 0;
	std::cerr << "GPU: " << gpu_id << std::endl;

	Json::Reader reader;
	Json::Value root;
	std::ifstream is;
	is.open("weight.json", std::ios::binary);
	if (reader.parse(is, root, false))
	{
	}
	else {
		is.close();
		std::cerr << "usage: weight.json" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	auto conv0_w = Read4DTensorFromRoot(root, "model.0.weight");
	auto conv3_w = Read4DTensorFromRoot(root, "model.3.weight");
	auto conv6_w = Read4DTensorFromRoot(root, "model.6.weight");
	auto classifier_weight = Read2DTensorFromRoot(root, "classifier.weight");
	auto classifier2_weight = Read2DTensorFromRoot(root, "classifier2.weight");

	is.close();
	vector<cv::Mat> inputs(10);
	cv::Mat image0 = load_image("0.png");
	cv::Mat image1 = load_image("1.png");
	cv::Mat image2 = load_image("2.png");
	cv::Mat image3 = load_image("3.png");
	cv::Mat image4 = load_image("4.png");
	cv::Mat image5 = load_image("5.png");
	cv::Mat image6 = load_image("6.png");
	cv::Mat image7 = load_image("7.png");
	cv::Mat image8 = load_image("8.png");
	cv::Mat image9 = load_image("9.png");
	inputs.push_back(image0);
	inputs.push_back(image1);
	inputs.push_back(image2);
	inputs.push_back(image3);
	inputs.push_back(image4);
	inputs.push_back(image5);
	inputs.push_back(image6);
	inputs.push_back(image7);
	inputs.push_back(image8);
	inputs.push_back(image9);
	cudaSetDevice(gpu_id);
	cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);
	/*int img_len = image.cols*image.rows * image.channels();
	float *input_data = new float[img_len];
	for (int i = 0; i < img_len; i++) {
		*(input_data + i) = image.at<float>(i);
	}

	Tensor input(image.cols, image.rows, 1, 1, input_data, false);*/
	Tensor input(image1.cols, image1.rows, 1, 10, inputs);

	cout << "gpu input node create successfully" << endl;
	Conv2D conv0(cudnn, conv0_w, 1, 6, 5, 5, 0);
	ActivateLayer relu1(cudnn);
	PoolingLayer maxpooling2(cudnn, 2);
	Conv2D conv3(cudnn, conv3_w, 6, 16, 5, 5, 0);
	ActivateLayer relu4(cudnn);
	PoolingLayer maxpooling5(cudnn, 2);
	Conv2D conv6(cudnn, conv6_w, 16, 120, 5, 5, 0);
	ActivateLayer relu7(cudnn);
	Dense classifier(cudnn, classifier_weight, 120, 84);
	Dense classifier2(cudnn, classifier2_weight, 84, 10);
	
	auto beginTime = std::chrono::high_resolution_clock::now();
	Tensor out0 = conv0(input/*, conv0_w*/);
	Tensor out1 = relu1(out0);
	Tensor out2 = maxpooling2(out1);
	Tensor out3 = conv3(out2/*, conv3_w*/);
	Tensor out4 = relu4(out3);
	Tensor out5 = maxpooling5(out4);
	Tensor out6 = conv6(out5/*, conv6_w*/);
	Tensor out7 = relu7(out6);
	Tensor out8 = classifier(out7/*, classifier_weight*/);
	Tensor out9 = classifier2(out8/*, classifier2_weight*/);
	auto endTime = std::chrono::high_resolution_clock::now();
	auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - beginTime);
	for (int i = 0; i < 10; i++) {
		cout << "the num is: " << argmax(vector<float>(out9.get_data() + i * 10, out9.get_data() + (i + 1) * 10)) << endl;
	}
	cout << "gpu calculate successfully: " << elapsedTime.count()<< "us" << endl;
	cudnnDestroy(cudnn);

	
	cout << "cpu input node create successfully" << endl;
	
	beginTime = std::chrono::high_resolution_clock::now();

	/*float *input_data_cpu = new float[img_len];
	for (int i = 0; i < img_len; i++) {
		*(input_data_cpu + i) = image.at<float>(i);
	}*/
	CPUTensor input_cpu(image1.cols, image1.rows, 1, 10, inputs);

	CPUConv2D conv0_cpu(1, 6, 5, 5, 0);
	CPUTensor out0_cpu = conv0_cpu(input_cpu, conv0_w);

	CPUActivateLayer relu1_cpu;
	CPUTensor out1_cpu = relu1_cpu(out0_cpu);

	CPUPoolingLayer maxpooling2_cpu(2);
	CPUTensor out2_cpu = maxpooling2_cpu(out1_cpu);

	CPUConv2D conv3_cpu(6, 16, 5, 5, 0);
	CPUTensor out3_cpu = conv3_cpu(out2_cpu, conv3_w);

	CPUActivateLayer relu4_cpu;
	CPUTensor out4_cpu = relu4_cpu(out3_cpu);

	CPUPoolingLayer maxpooling5_cpu(2);
	CPUTensor out5_cpu = maxpooling5_cpu(out4_cpu);

	CPUConv2D conv6_cpu(16, 120, 5, 5, 0);
	CPUTensor out6_cpu = conv6_cpu(out5_cpu, conv6_w);

	CPUActivateLayer relu7_cpu;
	CPUTensor out7_cpu = relu7_cpu(out6_cpu);

	CPUDense classifier_cpu(120, 84);
	CPUTensor out8_cpu = classifier_cpu(out7_cpu, classifier_weight);

	CPUDense classifier2_cpu(84, 10);
	CPUTensor out9_cpu = classifier2_cpu(out8_cpu, classifier2_weight);

	endTime = std::chrono::high_resolution_clock::now();
	elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - beginTime);
	for (int i = 0; i < 10; i++) {
		cout << "the num is: " << argmax(vector<float>(out9_cpu.data + i * 10, out9_cpu.data + (i + 1) * 10)) << endl;
	}

	cout << "cpu conv calculate successfully: " << elapsedTime.count() << "us" << endl;
}