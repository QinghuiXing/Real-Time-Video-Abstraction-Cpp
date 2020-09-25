#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

namespace abst {
	class abstraction
	{
	public:
		abstraction();
		~abstraction();

		// step 2
		// Bilateral filter
		int s = -1;
		double sigmacolor = 4.25;
		double sigmaspace = 3;

		// step 3
		// Color quantization
		float phieq = 1;

		// step 5
		float sigma = 1;
		float phie = 5; // 0.75 to 5
		float tau = 0.98; //

		float qnearest(float in);

		float qnearest_10(float in);

		std::vector<float> gauss2D(unsigned * shape, float sigma);
		
		cv::Mat lab2bgr(cv::Mat &L, cv::Mat &a, cv::Mat &b);

		void bgr2lab_Labvalue(cv::Mat& bgr, cv::Mat &finalLab);

		cv::Mat lab2bgr_notsave(cv::Mat &L, cv::Mat &a, cv::Mat &b);

		void main(const char * filename);
	};

}

