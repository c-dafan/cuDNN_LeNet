#include"util.h"

cv::Mat load_image(const char* path) {
	auto img = cv::imread(path);
	cv::cvtColor(img, img, CV_BGR2GRAY);
	cv::imshow("aa", img);
	cv::waitKey(500);
	cv::destroyAllWindows();
	cv::Mat out_img;
	cv::resize(img, out_img, cv::Size(32, 32));
	out_img.convertTo(out_img, CV_32FC1);
	return out_img;
}

void save_image(char *name, cv::Mat out_img) {
	cv::imwrite(string(name), out_img);
	cv::imshow(string(name), out_img);
	cv::waitKey(5000);
	cv::destroyAllWindows();
}

int argmax(const vector<float>& softmax) {
	int b = softmax.at(0);
	int a = 0;
	for (auto it = softmax.begin(); it != softmax.end(); it++) {
		if (*it > b) {
			b = *it;
			a = it - softmax.begin();
		}
	}
	return a;
}

Conv_W Read4DTensorFromRoot(const Json::Value& root, const char* key)
{
	auto tmp_data = root[key];
	int N = tmp_data.size();
	int C = 0;
	int W = 0;
	int H = 0;
	vector<vector<vector<vector<float>>>> Data(N);
	for (int n = 0; n < N; ++n)  // ±éÀúÊý×é  
	{
		C = tmp_data[n].size();
		vector<vector<vector<float>>> CData(C);
		for (int c = 0; c < C; c++) {
			W = tmp_data[n][c].size();
			vector<vector<float>> WData(W);
			for (int w = 0; w < W; w++) {
				H = tmp_data[n][c][w].size();
				vector<float> HData(H);
				for (int h = 0; h < H; h++) {
					HData[h] = tmp_data[n][c][w][h].asFloat();
				}
				WData[w] = HData;
			}
			CData[c] = WData;
		}
		Data[n] = CData;
	}
	return Conv_W(std::move(Data), N, C, W, H);
}

Dense_W Read2DTensorFromRoot(const Json::Value& root, const char* key)
{
	auto tmp_data = root[key];
	int N = tmp_data.size();
	vector<vector<float>> Data(N);
	int H = 0;
	for (int w = 0; w < N; w++) {
		H = tmp_data[w].size();
		vector<float> HData(H);
		for (int h = 0; h < H; h++) {
			HData[h] = tmp_data[w][h].asFloat();
		}
		Data[w] = HData;
	}
	return Dense_W(std::move(Data), N, H);
}


Conv_W::Conv_W(vector<vector<vector<vector<float>>>>data, int out, int in, int w, int h) :data(data), n(out), c(in), w(w), h(h)
{
}

Conv_W::~Conv_W()
{
}

Dense_W::Dense_W(vector<vector<float>> data, int out, int in) :data(data), n(out), c(in)
{
}

Dense_W::~Dense_W()
{
}