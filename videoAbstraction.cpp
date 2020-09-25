#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>

#include "abstraction.h"

using namespace cv;
using namespace std;


int main()
{
	abst::abstraction Abst;

	Mat src_bgr = cv::imread("imgs/7.jpg");

	/*****    Convert RGB Space to CIELAB Space    ****/

	int iterNum = 1;
	int iters = iterNum;

	auto s = std::chrono::high_resolution_clock::now();
	Mat finalImg(src_bgr.rows, src_bgr.cols, CV_8UC3);
while(iterNum--){
	
	unsigned rows = src_bgr.rows, cols = src_bgr.cols;

	Mat src_lab, src_Lab_vec[3];
	
	Abst.bgr2lab_Labvalue(src_bgr, src_lab);
	
	split(src_lab, src_Lab_vec);

	Mat L0 = src_Lab_vec[0].clone();


	/*     Bilateral Filtering L, a, b Channel Seperately      */

	Mat bf_tL(rows, cols, CV_32FC1, Scalar(1.0));
	Mat bf_ta(rows, cols, CV_32FC1, Scalar(1.0));
	Mat bf_tb(rows, cols, CV_32FC1, Scalar(1.0));
	Mat bf_L(rows, cols, CV_32FC1, Scalar(1.0));
	Mat bf_a(rows, cols, CV_32FC1, Scalar(1.0));
	Mat bf_b(rows, cols, CV_32FC1, Scalar(1.0));
	
	bf_L = src_Lab_vec[0].clone();
	bf_a = src_Lab_vec[1].clone();
	bf_b = src_Lab_vec[2].clone();

	for (unsigned tmp_i = 0; tmp_i < 2; tmp_i++) {
		bilateralFilter(bf_L, bf_tL, Abst.s, Abst.sigmacolor, Abst.sigmaspace);
		bilateralFilter(bf_a, bf_ta, Abst.s, Abst.sigmacolor, Abst.sigmaspace);
		bilateralFilter(bf_b, bf_tb, Abst.s, Abst.sigmacolor, Abst.sigmaspace);
		
		bilateralFilter(bf_tL, bf_L, Abst.s, Abst.sigmacolor, Abst.sigmaspace);
		bilateralFilter(bf_ta, bf_a, Abst.s, Abst.sigmacolor, Abst.sigmaspace);
		bilateralFilter(bf_tb, bf_b, Abst.s, Abst.sigmacolor, Abst.sigmaspace);
	}

	//Mat bf_bgr = Abst.lab2bgr(bf_L, bf_a, bf_b);

	//imwrite("bilateral filtered image.jpg", bf_bgr);
	

	/*    Luminance Quantization    */

	Mat Quantum(rows, cols, CV_32FC1, Scalar(1.0));

	Point2i pos = Point(0, 0);
	for (; pos.y < rows; pos.y++) {
		pos.x = 0;
		for (; pos.x< cols; pos.x++) {
			float tmp = Abst.qnearest_10(bf_L.ptr<float>(pos.y)[pos.x]) +
				5 * tanh(
					(bf_L.ptr<float>(pos.y)[pos.x] - 
						Abst.qnearest_10(bf_L.ptr<float>(pos.y)[pos.x])) * 
					Abst.phieq
					);
			Quantum.ptr<float>(pos.y)[pos.x] = tmp;
		}
	}

	//Mat lq_bgr = Abst.lab2bgr(Quantum, bf_a, bf_b);
	//imwrite("color quantized image.jpg", lq_bgr);


	/*     DoG Edge Detection     */

	Mat imgE(rows, cols, CV_32FC1, Scalar(1.0));
	Mat imgR(rows, cols, CV_32FC1, Scalar(1.0));

	unsigned shape[] = { 5,5 };
	auto filter = Abst.gauss2D(shape, Abst.sigma);
	auto filter2 = Abst.gauss2D(shape, Abst.sigma*1.265);
	Mat filt = Mat(filter);
	Mat filt2 = Mat(filter2);
	
	filt = filt.reshape(0, shape[0]);
	filt2 = filt2.reshape(0, shape[0]);
	
	filter2D(L0, imgE, -1, filt);
	filter2D(L0, imgR, -1, filt2);

	Mat D(rows, cols, CV_32FC1, Scalar(1.0));
	
	pos.x = 0;
	pos.y = 0;
	for (; pos.y < rows; pos.y++) {
		pos.x = 0;
		for (; pos.x < cols; pos.x++) {
			if (imgE.ptr<float>(pos.y)[pos.x] - Abst.tau*imgR.ptr<float>(pos.y)[pos.x] > 0) {
				D.ptr<float>(pos.y)[pos.x] = 1.0;
			}
			else {
				D.ptr<float>(pos.y)[pos.x] = 1 +
					tanh((imgE.ptr<float>(pos.y)[pos.x] - Abst.tau*imgR.ptr<float>(pos.y)[pos.x])*Abst.phie);
			}
		}
	}
	//Mat edges(rows, cols, CV_8UC3);
	//D.convertTo(edges, CV_8UC3, 255);
	//imwrite("edges.jpg", edges);
	

	/*     Merge Edeg and Luminance      */

	Mat merge_bgr = Abst.lab2bgr_notsave(Quantum, bf_a, bf_b);
	Mat img3(rows, cols, CV_32FC3, Scalar(1.0));
	pos.x = 0;
	pos.y = 0;
	for (; pos.y < rows; pos.y++) {
		pos.x = 0;
		for (; pos.x < cols; pos.x++) {
			float r, g, b;
			if (D.ptr<float>(pos.y)[pos.x] < 0.1) {
				r = g = b = 255.0 * D.ptr<float>(pos.y)[pos.x];
			}
			else {
				b = 255.0 * merge_bgr.at<Vec3f>(pos.y,pos.x)[0];
				g = 255.0 * merge_bgr.at<Vec3f>(pos.y, pos.x)[1];
				r = 255.0 * merge_bgr.at<Vec3f>(pos.y, pos.x)[2];
			}
			img3.at<Vec3f>(pos.y, pos.x)[0] = b;
			img3.at<Vec3f>(pos.y, pos.x)[1] = g;
			img3.at<Vec3f>(pos.y, pos.x)[2] = r;
		}
	}

	
	img3.convertTo(finalImg, CV_8UC3);

	//imwrite("final.jpg", finalImg);
}
	
	auto e = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = e - s;
	printf("total time: %f seconds | per frame: %f seconds",time, time/float(iters));
	imshow("src", src_bgr);
	imshow("abstracted", finalImg);
	waitKey(0);
	imwrite("final.jpg", finalImg);
	return 0;
}