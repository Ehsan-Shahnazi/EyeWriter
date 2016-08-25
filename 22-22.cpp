#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/videoio/videoio.hpp>

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>

using namespace cv;
using namespace std;

// حداقل و حداکثر سایز نقاطی که باید بررسی شود. خارج از این محدوده بررسی نشود
double min_eye_size=85;
double max_eye_size=95;

Mat frame; //current video frame
Mat frame1;//face
Mat frame2;//eye
Mat frame3;//pupil
Mat framepupil;
Mat cropedImage;
Rect croppedRectangle;
Mat cropedImage2;
Rect croppedRectangle2;
Mat morph_close;
Mat morph_open;
Mat morph_grad;
vector<Vec3f> circles;
int radius;
int x=1;
int tx=1;
int y=1;
int ty=1;
void movemouse(Point &Centroid)
{

}

void mousemove(int x_pos, int y_pos)
{
    ///Strings that will contain the conversions
    string xcord; string ycord;

    ///These are buffers or something? I don't really know... lol.
    stringstream sstr; stringstream sstr2;

    ///Conversion to regular string happens here
    sstr<<5*x_pos;
    xcord = sstr.str();
    sstr2<<5*y_pos;
    ycord = sstr2.str();

    ///Getting the command string
    string command = "xdotool mousemove " + xcord + " " + ycord;

    ///Converting command string to a form that system() accepts.
    const char *com = command.c_str();
    system(com);
}


int main()
{
	CascadeClassifier face_cascade("/home/elinux/opencv-3.1.0/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml");
	CascadeClassifier eye_cascade("/home/elinux/opencv-3.1.0/data/haarcascades_cuda/haarcascade_righteye_2splits.xml");

	// Open the video file
	VideoCapture capture(0);
	if (!capture.isOpened())
		return 1;

	bool stop(false);

	namedWindow("Extracted Frame",WINDOW_AUTOSIZE);
	int delay = 1;

	while (!stop)
	{
		if (!capture.read(frame))
			break;

		 // یافتن صورت
		 vector<Rect> fce;
		 face_cascade.detectMultiScale( frame, fce,
				 1.1, 2, 0|CASCADE_SCALE_IMAGE,
				 Size(300, 300),//حدقل انداز که باید بررسی شود
				 Size(1000, 1000) );//جداکثر امدازه که باید بررسی شود


 for( int i = 0; i < fce.size(); i++ )
	 {
		rectangle(frame, fce[i], Scalar(0, 255, 0),3);
		croppedRectangle=Rect(fce[i].x,fce[i].y,fce[i].width /2,fce[i].height/2);
		cropedImage = frame(croppedRectangle);
	//	imshow ("cropedImage",cropedImage);//سمت چپ صورت

		 // یافتن چشم ها
		 frame2=cropedImage;
		 vector<Rect> eyes;
		 eye_cascade.detectMultiScale( frame2, eyes,
				 1.1, 2, 0|CASCADE_SCALE_IMAGE,
				 Size(min_eye_size, min_eye_size),//حدقل انداز که باید بررسی شود
				 Size(max_eye_size, max_eye_size) );//جداکثر امدازه که باید بررسی شود

		 for( int t = 0; t < eyes.size(); t++ )
		 {
				rectangle(frame2, eyes[t], Scalar(0, 255, 0),3);
				croppedRectangle2=Rect(eyes[t].x,eyes[t].y+40,eyes[t].width ,eyes[t].height-40);
				cropedImage2 = frame2(croppedRectangle2);
		//		imshow ("cropedImage2",cropedImage2);//چشم


				//یافتن مردمک
				frame3=cropedImage2;
						cvtColor( frame3, frame3, COLOR_BGR2GRAY );

						frame3.convertTo(framepupil,//افزایش کنتراست
									   -1,//عمق تصویر خروجی-اگر منفی باشد همانند تصویر ورودی خواهد شد
									   1,//عامل ضرب؛ ارزش هر پیکسل در این عدد ضرب خواهد شد
									   0); //این مقدار به پیکسهای ضرب شده اضافه می شود-هر چه بیشتر شود،تصویر سفیدتر می شود

					//	framepupil=~framepupil;

							////opening & closing
						Mat se(5, 5, CV_8U, Scalar(1));

						se.at<uchar>(0, 0) = 0;
						se.at<uchar>(0, 1) = 0;
						se.at<uchar>(0, 3) = 0;
						se.at<uchar>(0, 4) = 0;

						se.at<uchar>(1, 0) = 0;
						se.at<uchar>(1, 4) = 0;

						se.at<uchar>(3, 0) = 0;
						se.at<uchar>(3, 4) = 0;

						se.at<uchar>(4, 0) = 0;
						se.at<uchar>(4, 1) = 0;
						se.at<uchar>(4, 3) = 0;
						se.at<uchar>(4, 4) = 0;


						//	morphologyEx(framepupil, morph_open, MORPH_OPEN,se,Point(-1,-1),1);
						//	morphologyEx(morph_open, morph_close, MORPH_CLOSE, Mat(), Point(-1, -1), 2);
					//	morphologyEx(framepupil, morph_grad, MORPH_GRADIENT,Mat(),Point(-1,-1),1);
						morphologyEx(framepupil, morph_grad, MORPH_GRADIENT,se,Point(-1,-1),1);


						  HoughCircles(morph_grad,//تصویر ورودی که باید سیاه و سفید باشد حتمن
								  circles,//متغییر برداری برای ذخیره سازی مقادیر xc,yc,r
								  HOUGH_GRADIENT,//روش تشخیص
								  1,//رزولوشن تصویر
								  1000,//حداقل فاصله بین مراکز شناسایی شده
						          100,//حداکثر آستانه برای الگوریتم لبه یابی کنی داخلی
								  15,//آستانه برای تشخیص مرکز
								  10,//حداقل شعاع برای شناسایی دایره ها
								  20); // حداکثر شعاع برای شناسایی دایره ها


							// رسم دایره دور مردمک
								vector<Vec3f>::const_iterator itc = circles.begin();

								while (itc != circles.end()) {

									circle(morph_grad,
										Point((*itc)[0], (*itc)[1]), // circle centre
										(*itc)[2], // circle radius اگر مقداربالا قرار دهیم یک نقطه روی مردمک ظاهر می شود
										Scalar(255), // color
										5); // thickness
										++itc;	}

						         Point center((circles[t][0]),(circles[t][1]));		//بعد از سنتر مساوی نمی خواهد
						         //	 radius = (circles[t][2]);

						         cout << "Circles: " << circles.size() << endl;
						         cout << "center " << center << endl;
						      //   cout << "(circles[t][0]) " << (circles[t][0]) << endl;
						      //   cout << "(circles[t][1]) " << (circles[t][1]) << endl;
					        // cout << "اندازه شعاع" << radius << endl;

								if(circles.size()!=0){

									(circles[t][0])=(circles[t][0])*((circles[t][0])/10);
									(circles[t][1])=(circles[t][1])*((circles[t][1])/10);

								/*	(circles[t][0])=(circles[t][0])*8;
									(circles[t][1])=(circles[t][1])*8;/*
								/*	for(int z=0;z<1;z++){
										xx=(circles[t][0]);
										yy=(circles[t][1]);	}

									if(xx>(circles[t][0])){
										xx=(circles[t][0]);
										(circles[t][0])=(circles[t][0])*10;}

									if (xx<(circles[t][0])){
										xx=(circles[t][0]);
										(circles[t][0])=(circles[t][0])/10;}

									if(yy>(circles[t][1])){
										yy=(circles[t][0]);
										(circles[t][1])=(circles[t][1])*10;}

									if (yy<(circles[t][1])){
										yy=(circles[t][0]);
										(circles[t][1])=(circles[t][1])/10;}

									if ((circles[t][0])<0)(circles[t][0])=0;
									if ((circles[t][1])<0)(circles[t][1])=0;/*

							/*	if((circles[t][0])>(circles[t][0])){
								(circles[t][0])=(circles[t][0])*10;}

								if ((circles[t][0])<(circles[t][0])){
									(circles[t][0])=(circles[t][0])/10;}

								if((circles[t][1])>(circles[t][1])){
									(circles[t][1])=(circles[t][1])*10;}

								if ((circles[t][1])<(circles[t][1])){
								(circles[t][1])=(circles[t][1])/10;}*/




						         mousemove((circles[t][0]),(circles[t][1]));
		cout << "(circles[t][0]) , (circles[t][1]) " << (circles[t][0]) <<","<< (circles[t][1]) << endl;

//xx=(circles[t][0]);
//yy=(circles[t][1]);
								}

							//	imshow ("framepupil",framepupil);
							//	imshow ("morph_open",morph_open);
							//	imshow ("morph_colse",morph_close);
								imshow ("morph_grad",morph_grad);

		 }}

   // mousemove(center);

		imshow("Extracted Frame", frame);
		if (waitKey(delay) >= 0)
			stop = true;
	}

	capture.release();
	waitKey();
}





