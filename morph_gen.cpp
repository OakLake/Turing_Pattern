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
void calc_reagnet(Mat,Mat,Mat,int,int);
void make_sym(Mat);
double my_rand();
void add_noise(Mat,int,double);
void calc_scale(Mat,Mat,Mat,int);
void psy_reagent(Mat,Mat,Mat,int,int,int);

///////////////////////

int main(){
  //
  srand(time(NULL));
  int seed = 1851156740;//rand(); //
  theRNG().state = seed;
  cout << "Random Seed: " << seed << endl;
  //
  Size size(600,600);
  Mat frame(size,CV_8SC1);
  randu(frame,Scalar(-127),Scalar(127));
  // circle(frame, Point(300,300), 200,Scalar(-42), -1);
  //
  //
  int num_scales = 2;
  vector<Mat> activ_vec;
  vector<Mat> inhib_vec;
  vector<Mat> indv_scale;
  for(int v=0;v<num_scales;v++){
    activ_vec.push_back(Mat::zeros(size,CV_8SC1));
    inhib_vec.push_back(Mat::zeros(size,CV_8SC1));
    indv_scale.push_back(Mat::zeros(size,CV_8SC1));
  }
  Mat global_chnge = Mat::zeros(size,CV_8SC1); // how to change
  Mat global_activ = Mat::zeros(size,CV_8SC1); // merger of all
  Mat global_inhib = Mat::zeros(size,CV_8SC1);
  //
  // ONLY ODD Nums & N[*] > P[*] Square kernels/scales
  int P[] = {13,17,23,43};
  int N[] = {23,23,43,73};
  int weights[] = {1,1,-1,4};

  int rate = 1;
  //
  int loops = 5000;
  int make_sym_FLAG = 1;
  int Method_FLAG = 1;


  namedWindow("Frame",WINDOW_AUTOSIZE);
  // moveWindow("Frame",1000,400);
  imshow("Frame",frame);
  waitKey(20);


  for(int i=0;i<loops;i++){

    cout << "\r*Iteration: " << i << flush;

    // threads are manually set, more than 4 is not beneficial
    thread t1(psy_reagent,frame, activ_vec[0],inhib_vec[0], P[0], N[0],-100);
    thread t2(psy_reagent,frame, activ_vec[1],inhib_vec[1], P[1], N[1],0);
    // thread t3(psy_reagent,frame, activ_vec[2],inhib_vec[2], P[2], N[2],-40);
    // thread t4(psy_reagent,frame, activ_vec[3],inhib_vec[3], P[3], N[3],10);

    t1.join();
    t2.join();
    // t3.join();
    // t4.join();


    if (make_sym_FLAG == 1){
      for(int s=0;s<num_scales;s++){
        make_sym(activ_vec[s]);
        make_sym(inhib_vec[s]);
      }
    }


    if (Method_FLAG == 1){
      for (int c=0;c<num_scales;c++){

        addWeighted(global_activ, 1, activ_vec[c], weights[c],0,global_activ);
        addWeighted(global_inhib, 1, inhib_vec[c], weights[c],0,global_inhib);
      }
      calc_scale(global_activ,global_inhib,global_chnge,rate);
    }else{
      for (int c=0;c<num_scales;c++){
        calc_scale(activ_vec[c],inhib_vec[c],indv_scale[c],rate);
        addWeighted(global_chnge, 1, indv_scale[c], weights[c],0,global_chnge);
      }
    }


    addWeighted(frame, 1, global_chnge, 1,0,frame);
    normalize(frame,  frame, -127, 127, NORM_MINMAX);

    imshow("Frame",frame);


    global_chnge = 0;
    global_activ = 0;
    global_inhib = 0;


    if(waitKey(50) >= 0){
      Mat wFrame;
      // normalize(frame,  wFrame, 0, 255, NORM_MINMAX);
      frame.convertTo(wFrame, CV_8UC1, 255.0);
      imwrite("./imgs/img_"+to_string(seed)+".png",wFrame);
      cout << endl << "Pattern Saved" << endl;
      break;
    }

  }

  return 0;
}



////////////// FUNCTIONS FUNCTIONS

int my_remap(int val,int H, int L,int H2,int L2){
  return (L2 + (val - L) * (H2 - L2) / (H - L));
}


void calc_reagnet(Mat src, Mat activ,Mat inhib, int p, int n){
  Mat padded_Mat = src.clone();
  int pad = n/2;
  copyMakeBorder( src, padded_Mat, pad, pad, pad, pad, BORDER_REPLICATE);

  int w_p = p/2;
  int w_n = n/2;
  int activ_conc,inhib_conc;
  // pixVal,

  for (int row=pad;row<src.rows+pad;row++){

    schar* ac = activ.ptr<schar>(row+1-pad);
    schar* nh = inhib.ptr<schar>(row+1-pad);

    for(int col=pad;col<src.cols+pad;col++){


      activ_conc = mean(padded_Mat(Rect(col-w_p,row-w_p,p,p)) )[0];
      inhib_conc = mean(padded_Mat(Rect(col-w_n,row-w_n,n,n)) )[0];

      *ac++ = activ_conc;
      *nh++ = inhib_conc;

    }
  }



}

void calc_scale(Mat activ,Mat inhib,Mat chnge,int rate){

  for(int r=0;r<chnge.rows;++r){
    schar* ac = activ.ptr<schar>(r);
    schar* nh = inhib.ptr<schar>(r);
    schar* ng = chnge.ptr<schar>(r);

    for (int c=0;c<chnge.cols;++c){
      if(*ac++ >= *nh++){
        *ng++ = rate;
      }else{
        *ng++ = -rate;
      }
    }
  }

  }

void psy_reagent(Mat src,Mat activ,Mat inhib,int p,int n,int thr){

  Mat padded_Mat = src.clone();
  int pad = n/2;
  copyMakeBorder( src, padded_Mat, pad, pad, pad, pad, BORDER_REPLICATE);

  int w_p = p/2;
  int w_n = n/2;
  int activ_conc,inhib_conc;
  // pixVal,

  for (int row=pad;row<src.rows+pad;row++){

    schar* ac = activ.ptr<schar>(row+1-pad);
    schar* nh = inhib.ptr<schar>(row+1-pad);

    for(int col=pad;col<src.cols+pad;col++){


      activ_conc = mean(padded_Mat(Rect(col-w_p,row-w_p,p,p)) )[0];
      inhib_conc = mean(padded_Mat(Rect(col-w_n,row-w_n,n,n)) )[0];

      if (activ_conc < thr){
        activ_conc = 0;
        inhib_conc = 0;
      }


      *ac++ = activ_conc;
      *nh++ = inhib_conc;

    }
  }
}

void make_sym(Mat scale){
  Mat clone = scale.clone();
  int nRow = clone.rows;
  int nCols = clone.cols;

  for(int i=0;i<nRow/2;i++){
    schar* pix_r = scale.ptr<schar>(i+1);
    schar* pix_rp = scale.ptr<schar>(nRow-i-1);


    for(int j=0;j<nCols/2;j++){

      int avg = 0.25*(*(pix_r+j) + *(pix_r+(nCols-j)) + *(pix_rp+j) + *(pix_rp+(nCols-j))  );
      *(pix_r+j) = avg;
      *(pix_r+(nCols-j)) =avg;
      *(pix_rp+j) = avg;
      *(pix_rp+(nCols-j)) = avg;
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
