#pragma once
#include<iostream>
#include<json/json.h>
#include<iostream>
#include<fstream>
#include<vector>
#include<opencv2\opencv.hpp>
#include<opencv2\core.hpp>
#include<opencv2\highgui.hpp>
using namespace std;

#define checkCUDNN(expression)                                  \
{                                                             \
	cudnnStatus_t status = (expression);                        \
	if (status != CUDNN_STATUS_SUCCESS) {                     \
		std::cerr << "Error on line " << __LINE__ << ": "       \
		<< cudnnGetErrorString(status) << std::endl;            \
		std::exit(EXIT_FAILURE);                                \
    }                                                        \
}
cv::Mat load_image(const char* path);
int argmax(const vector<float>& softmax);
void save_image(char *name, cv::Mat out_img);
class Conv_W
{
public:
	Conv_W(vector<vector<vector<vector<float>>>>data, int n, int c, int w, int h);
	~Conv_W();
	vector<vector<vector<vector<float>>>> data;
	int n, c, w, h;
};

class Dense_W
{
public:
	Dense_W(vector<vector<float>> data, int out, int in);
	~Dense_W();
	vector<vector<float>> data;
	int n, c;
};


Conv_W Read4DTensorFromRoot(const Json::Value& root, const char* key);

Dense_W Read2DTensorFromRoot(const Json::Value& root, const char* key);


