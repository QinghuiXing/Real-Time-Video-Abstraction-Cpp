#include "abstraction.h"
#include <cassert>
#include<cfloat>
#include<opencv2/opencv.hpp>
#include<iostream>
namespace abst {
	abstraction::abstraction()
	{
	}


	abstraction::~abstraction()
	{
	}





	float abstraction::qnearest(float in)
	{
		assert(in >= 0);
		
		if (in < 6.25)
			return 0;
		else if (in < (12.5 + 6.25))
			return 12.5;
		else if (in < (25 + 6.25))
			return 25;
		else if (in < (37.5 + 6.25))
			return 37.5;
		else if (in < (50 + 6.25))
			return 50;
		else if (in < (62.5 + 6.25))
			return 62.5;
		else if (in < (75 + 6.25))
			return 75;
		else if (in < (87.5 + 6.25))
			return 87.5;
		else
			return 100;

	}

	float abstraction::qnearest_10(float in)
	{
		assert(in >= 0);

		if (in < 5)
			return 0;
		else if (in < 15)
			return 10;
		else if (in < 25)
			return 20;
		else if (in < 35)
			return 30;
		else if (in < 45)
			return 40;
		else if (in < 55)
			return 50;
		else if (in < 65)
			return 60;
		else if (in < 75)
			return 70;
		else if (in < 85)
			return 80;
		else if (in < 95)
			return 90;
		else
			return 100;
	}

	std::vector<float> abstraction::gauss2D(unsigned * shape, float sigma)
	{
		float m = (float(shape[0]) - 1.) / 2.;
		float n = (float(shape[1]) - 1.) / 2.;
		float mm = m, nn = n;
		std::vector<float> x(shape[1]), y(shape[0]);
		for (int i=0; -m < mm + 1;i++, m--)y[i]=(m*m);
		for (int i=0; -n < nn + 1;i++, n--)x[i]=(n*n);
		std::vector<float> h(shape[0] * shape[1]);

		float max = 0;
		for (int i = 0; i < y.size(); i++) {
			for (int j = 0; j < x.size(); j++) {
				float tmpx = -(x[j] + y[i]) / (2.*sigma*sigma);
				tmpx = exp(tmpx);
				if (tmpx > max)
					max = tmpx;
				h[i*x.size()+j] = tmpx;
			}
		}
		float sumh = 0;
		for (int i = 0; i < h.size(); i++) {
			if (h[i] < FLT_MIN*max)
				h[i] = 0;
			sumh += h[i];
		}

		if (sumh == 0)return h;
		for (int i = 0; i < h.size(); i++) {
			h[i] /= sumh;
		}
		return h;

	}

	cv::Mat abstraction::lab2bgr(cv::Mat &L, cv::Mat &a, cv::Mat &b)
	{
		int row = L.rows, col = L.cols;

		std::vector<cv::Mat> labvec;
		labvec.push_back(L);
		labvec.push_back(a);
		labvec.push_back(b);

		cv::Mat lab(row, col, CV_32FC3, cv::Scalar(1.0, 1.0, 1.0));
		cv::merge(labvec, lab);

		cv::Mat bgr(row, col, CV_32FC3, cv::Scalar(1.0, 1.0, 1.0));
		cv::cvtColor(lab, bgr, cv::COLOR_Lab2BGR);

		bgr *= 255.0;

		cv::Mat bgr_save(row, col, CV_8UC3);
		bgr.convertTo(bgr_save, CV_8UC3);

		return bgr_save;
	}

	void abstraction::bgr2lab_Labvalue(cv::Mat &bgr, cv::Mat &Lab)
	{
		int rows = bgr.rows;
		int cols = bgr.cols;

		cv::Mat tmpLab(rows, cols, CV_32FC1, cv::Scalar(1.0));
		cv::cvtColor(bgr, tmpLab, cv::COLOR_BGR2Lab);
		
		std::vector<cv::Mat> vecs;
		cv::split(tmpLab, vecs);

		cv::Mat L(rows, cols, CV_32FC1, cv::Scalar(1.0));
		cv::Mat a(rows, cols, CV_32FC1, cv::Scalar(1.0));
		cv::Mat b(rows, cols, CV_32FC1, cv::Scalar(1.0));

		vecs[0].convertTo(L, CV_32FC1, 100.0 / 255.0);
		vecs[1].convertTo(a, CV_32FC1, 1.0);
		vecs[2].convertTo(b, CV_32FC1, 1.0);

		a -= 128;
		b -= 128;

		std::vector<cv::Mat>finalVecs;
		cv::Mat finalLab;
		finalVecs.push_back(L);
		finalVecs.push_back(a);
		finalVecs.push_back(b);

		cv::merge(finalVecs, finalLab);
		Lab = finalLab.clone();
	}

	cv::Mat abstraction::lab2bgr_notsave(cv::Mat & L, cv::Mat & a, cv::Mat & b)
	{
		int row = L.rows, col = L.cols;

		std::vector<cv::Mat> labvec;
		labvec.push_back(L);
		labvec.push_back(a);
		labvec.push_back(b);

		cv::Mat lab(row, col, CV_32FC3, cv::Scalar(1.0, 1.0, 1.0));
		cv::merge(labvec, lab);

		cv::Mat bgr(row, col, CV_32FC3, cv::Scalar(1.0, 1.0, 1.0));
		cv::cvtColor(lab, bgr, cv::COLOR_Lab2BGR);

		return bgr;
	}

	void abstraction::main(const char * filename)
	{


	}


}


