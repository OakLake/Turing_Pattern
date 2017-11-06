// Morphogenesis
// Sammy Hasan
// 2017


// btw OpenCV Mat are smartly passed.
// OpenCV HSV : H: 0 - 180, S: 0 - 255, V: 0 - 255
//  Hue, Saturation, Value

# include <opencv2/opencv.hpp>
# include <opencv2/highgui/highgui.hpp>
# include <iostream>
# include <stdlib.h>
# include <string>
# include <cstdlib>
# include <thread>
# include <random>
# include <cmath>
# include <tuple>

using namespace std;
using namespace cv;

int my_remap(int,int,int,int,int);
void apply_scale(Mat,Mat,int,int,int);
void make_sym(Mat);
double my_rand();
void add_noise(Mat,int,double);



///////////////////////

int main(){
  //
  srand(time(NULL));
  int seed = rand();
  theRNG().state = seed;
  cout << "Random Seed: " << seed << endl;
  //
  Size size(500,500);
  Mat frame(size,CV_8SC1);
  randu(frame,Scalar(-127),Scalar(127));
  Mat display,displayRGB;
  //
  //
  int num_scales = 2;
  vector<Mat> chng_vec;
  for(int v=0;v<num_scales;v++){
    chng_vec.push_back(Mat::zeros(size,CV_8SC1));
  }
  Mat global_chnge = Mat::zeros(size,CV_8SC1); // merger of all
  //
  // ONLY ODD Nums & N[*] > P[*] Square kernels/scales
  int P[] = {13,41};
  int N[] = {21,71};
  int weights[] = {-8,1};
  //
  int loops = 5000;
  int make_sym_FLAG = 0;


  namedWindow("Frame",WINDOW_AUTOSIZE);
  imshow("Frame",frame);
  waitKey(1);


  for(int i=0;i<loops;i++){

    cout << "\r Iteration: " << i << flush;

    // reset change imgs to zero
    for(Mat chnge : chng_vec){
      chnge = 0;
    }

    // threads are manually set, more than 4 is not beneficial
    thread t1(apply_scale,frame, chng_vec[0], P[0], N[0],-127);
    thread t2(apply_scale,frame, chng_vec[1], P[1], N[1],0);


    t1.join();
    t2.join();

    if (make_sym_FLAG == 1){
      for (Mat chnge : chng_vec){
        make_sym(chnge);
      }
    }


    for (int c=0;c<chng_vec.size();c++){
      addWeighted(global_chnge, 1, chng_vec[c], weights[c],0,global_chnge);
    }
    addWeighted(frame, 1, global_chnge, 1,0,frame);
    normalize(frame,  frame, -127, 127, NORM_MINMAX);

    // add_noise(frame,90,0.95);

    // write_f.convertTo(write_f, CV_8UC1, 255.0);

    imshow("Frame",frame);

    // if(i%10==0)
    // {
      // Mat write;
      // normalize(display,  write, 0, 255, NORM_MINMAX);
      // display.convertTo(write, CV_8UC1, 255.0);
      // string PATH = "./video2/";
      // imwrite(PATH + "img_S"+to_string(seed)+ "_i_"+ to_string(i)+".png",write);
    // }

    if(waitKey(1) >= 0){
      break;
    }

  }

  cout << endl;



  return 0;
}



////////////// FUNCTIONS FUNCTIONS

int my_remap(int val,int H, int L,int H2,int L2){
  return (L2 + (val - L) * (H2 - L2) / (H - L));
}


void apply_scale(Mat src, Mat dst, int p, int n,int thr){
  Mat padded_Mat = src.clone();
  int pad = n/2;
  copyMakeBorder( src, padded_Mat, pad, pad, pad, pad, BORDER_REPLICATE);

  int w_p = p/2;
  int w_n = n/2;

  for (int row=pad;row<src.rows+pad;row++){
    for(int col=pad;col<src.cols+pad;col++){
      double p_scale = mean(padded_Mat(Rect(col-w_p,row-w_p,p,p)))[0];
      double n_scale = mean(padded_Mat(Rect(col-w_n,row-w_n,n,n)))[0];
      int diff = (p_scale - n_scale) / abs(p_scale - n_scale);
      dst.at<uchar>(row-pad,col-pad) = diff;


      // if (p_scale >= thr){
      //
      // }
    }
  }



}


void make_sym(Mat scale){
  Mat clone = scale.clone();
  for(int i=0;i<clone.rows/2+1;i++){
    for(int j=0;j<clone.cols/2+1;j++){
      int avg = 0.25*(clone.at<uchar>(j,i) + clone.at<uchar>(j,clone.rows-i) + clone.at<uchar>(clone.cols-j) + clone.at<uchar>(clone.cols-j,clone.rows-i));
      scale.at<uchar>(j,i) = avg;
      scale.at<uchar>(j,scale.rows-i) = avg;
      scale.at<uchar>(scale.cols-j,i) = avg;
      scale.at<uchar>(scale.cols-j,scale.rows-i) = avg;
    }
  }
}


double my_rand(){
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);
  return dis(gen);
}


void add_noise(Mat orig,int amount,double prob){
  if (my_rand() >= prob){
    Mat noise;
    randu(noise,Scalar(-amount),Scalar(amount));
    orig = orig + noise;
  }

}
