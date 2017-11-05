// Morphogenesis
// Sammy Hasan
// 2017


# include <opencv2/opencv.hpp>
# include <opencv2/highgui/highgui.hpp>
# include <iostream>
# include <stdlib.h>
# include <string>
# include <cstdlib>
# include <thread>

using namespace std;
using namespace cv;


void apply_scale(Mat src, Mat dst, int p, int n){
  int pad = n/2+1;
  for (int row=pad;row<src.rows-pad;row++){
    for(int col=pad;col<src.cols-pad;col++){
      double p_scale = mean(src(Rect(row-p/2,col-p/2,p,p)))[0];
      double n_scale = mean(src(Rect(row-n/2,col-n/2,n,n)))[0];
      dst.at<uchar>(col,row) = (p_scale - n_scale) / abs(p_scale-n_scale);
    }
  }
}


int main(){

  srand(time(NULL));
  int seed = rand();
  theRNG().state = seed;
  cout << "Random Seed: " << seed << endl;
  Size size(800,800);


  Mat frame(size,CV_8SC1);
  randu(frame,Scalar(-127),Scalar(127));

  Mat change_img = Mat::zeros(size,CV_8SC1);
  Mat change_img2 = Mat::zeros(size,CV_8SC1);
  Mat change_img3 = Mat::zeros(size,CV_8SC1);

  int p = 80;
  int n = 120;

  int p2 = 15;
  int n2 = 25;

  int pad = n/2+1;


  namedWindow("Frame",WINDOW_AUTOSIZE);
  imshow("Frame",frame(Rect(pad,pad,frame.rows-2*pad,frame.cols-2*pad)));
  waitKey(0);

  // Morph
  cout << "Press Enter To Begin!" << endl;
  int loops = 1000;
  // VideoWriter outputVideo;  // Open the output
  // outputVideo.open("./vid.avi", -1, 10, size, true);

  for(int i=0;i<loops;i++){
    cout << "\r Iteration: " << i << flush;
    change_img = 0;
    change_img2 = 0;
    change_img3 = 0;

    thread t1(apply_scale,frame, change_img, p, n);
    thread t2(apply_scale,frame, change_img2, p2, n2);
    thread t3(apply_scale,frame, change_img3, 5, 10);
    t1.join();
    t2.join();
    t3.join();
    // apply_scale(frame, change_img, p, n);
    // apply_scale(frame, change_img2, p2, n2);

    addWeighted(change_img, 2, change_img2, 1,0,change_img);
    addWeighted(change_img, 1, change_img3, 3,0,change_img);
    addWeighted(frame, 1, change_img, 1,0,frame);

    normalize(frame,  frame, -127, 127, NORM_MINMAX);
    // write_f = frame(Rect(pad,pad,frame.rows-pad,frame.cols-pad)).clone();
    // write_f.convertTo(write_f, CV_8UC1, 255.0);
    // outputVideo.write(frame);
    imshow("Frame",frame(Rect(pad,pad,frame.rows-2*pad,frame.cols-2*pad)));
    // imwrite("./video/img_"+to_string(i)+".jpg",write_f);

    if(waitKey(5) >= 0){
      break;
    }

  }
  cout << endl;



  return 0;
}
