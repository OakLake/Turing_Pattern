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
// void mirror_pad(Mat,Mat,int);
///////////////////////

int main(){
  //
  srand(time(NULL));
  int seed = rand();
  theRNG().state = seed;
  cout << "Random Seed: " << seed << endl;
  //
  Size size(500,500);
  Size size_pad(600,600);
  Mat frame(size_pad,CV_8SC1);
  randu(frame,Scalar(-127),Scalar(127));
  // circle(frame, Point(300,300), 200,Scalar(-42), -1);
  //
  //
  int num_scales = 3;
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
  int P[] = {3,11,23,43};
  int N[] = {11,79,43,73};
  int weights[] = {20,3,1,4};
  int thr[] = {-100,-10,42};
  int rate[] = {-10,40,50};
  int global_rate = 1;
  //
  int loops = 5000;
  int make_sym_FLAG = 1;
  int Method_FLAG = 2;


  namedWindow("Frame",WINDOW_AUTOSIZE);
  // moveWindow("Frame",1000,400);
  imshow("Frame",frame);
  waitKey(20);


  for(int i=0;i<loops;i++){

    cout << "\r*Iteration: " << i << flush;

    // threads are manually set, more than 4 is not beneficial
    // thread t1(calc_reagnet,frame, activ_vec[0],inhib_vec[0], P[0], N[0]);
    // thread t2(calc_reagnet,frame, activ_vec[1],inhib_vec[1], P[1], N[1]);
    // thread t3(psy_reagent,frame, activ_vec[2],inhib_vec[2], P[2], N[2],-40);
    // thread t4(psy_reagent,frame, activ_vec[3],inhib_vec[3], P[3], N[3],10);

    // t1.join();
    // t2.join();
    // t3.join();
    // t4.join();

    for(int s=0;s<num_scales;s++){
      activ_vec[s] = 0;
      inhib_vec[s] = 0;
      for(int row=50;row<frame.rows-50;row++){
        for(int col=50;col<frame.cols-50;col++){
          int p = P[s];
          int n = N[s];
          int w_p = p/2;
          int w_n = n/2;
          double p_scale = mean(frame(Rect(row-w_p,col-w_p,p,p)) )[0];
          if (p_scale >= thr[s]){
            double n_scale = mean(frame(Rect(row-w_n,col-w_n,n,n)) )[0];
            activ_vec[s].at<schar>(row-45,col-45) = p_scale;
            inhib_vec[s].at<schar>(row-45,col-45) = n_scale;
          }

        }
      }
    }




    // cout << "No Error" << endl;

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
      calc_scale(global_activ,global_inhib,global_chnge,global_rate);
    }else{
      for (int c=0;c<num_scales;c++){
        calc_scale(activ_vec[c],inhib_vec[c],indv_scale[c],rate[c]);
        addWeighted(global_chnge, 1, indv_scale[c], weights[c],0,global_chnge);
      }
    }


    addWeighted(frame(Rect(50,50,500,500)), 1, global_chnge, 1,0,frame(Rect(50,50,500,500)));
    normalize(frame,  frame, -127, 127, NORM_MINMAX);

    imshow("Frame",frame);
    add_noise(frame(Rect(50,50,500,500)),10,0.7);

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
  int pad = 50;//n/2;
  // copyMakeBorder( src, padded_Mat, pad, pad, pad, pad, BORDER_REPLICATE);

  int w_p = p/2;
  int w_n = n/2;
  int pixVal,activ_conc,inhib_conc;


  for (int row=pad;row<src.rows+pad;row++){
    for(int col=pad;col<src.cols+pad;col++){


      pixVal = padded_Mat.at<schar>(col,row);

      activ_conc = mean(padded_Mat(Rect(col-w_p,row-w_p,p,p)) )[0];
      if (activ_conc >= 0){
        inhib_conc = mean(padded_Mat(Rect(col-w_n,row-w_n,n,n)) )[0];

        activ.at<schar>(row-pad,col-pad) = activ_conc;
        inhib.at<schar>(row-pad,col-pad) = inhib_conc;
      }


    }
  }

// Mat random_pad(Mat src,int pad){
//   Mat dst(src.cols+pad,src.rows+pad,CV_8SC1);
//   randu(dst,Scalar(-127),Scalar(127));
//   return dst;
// }
//
// Mat unpad(Mat src,int pad){
//
// }
  // Mat padded_Mat = src.clone();
  // int pad = n/2;
  //
  // // copyMakeBorder( src, padded_Mat, pad, pad, pad, pad, BORDER_CONSTANT,Scalar(rand()));
  //
  // int w_p = p/2;
  // int w_n = n/2;
  // int activ_conc,inhib_conc;
  // // pixVal,
  //
  // for (int row=pad;row<src.rows-pad;row++){
  //
  //   schar* ac = activ.ptr<schar>(row+1-pad);
  //   schar* nh = inhib.ptr<schar>(row+1-pad);
  //
  //   for(int col=pad;col<src.cols-pad;col++){
  //
  //
  //     activ_conc = mean(padded_Mat(Rect(col-w_p,row-w_p,p,p)) )[0];
  //     inhib_conc = mean(padded_Mat(Rect(col-w_n,row-w_n,n,n)) )[0];
  //
  //     *ac++ = activ_conc;
  //     *nh++ = inhib_conc;
  //
  //   }
  // }



}

void calc_scale(Mat activ,Mat inhib,Mat chnge,int rate){
  int ac,nh;
  for (int row=0;row<activ.rows;row++){
    for(int col=0;col<activ.cols;col++){
      ac = activ.at<schar>(row,col);
      nh = inhib.at<schar>(row,col);
      if (ac >= nh){
        chnge.at<schar>(row,col) = rate;
      }else{
        chnge.at<schar>(row,col) = -rate;
      }

      }
    }
  // for(int r=0;r<chnge.rows;++r){
  //   schar* ac = activ.ptr<schar>(r);
  //   schar* nh = inhib.ptr<schar>(r);
  //   schar* ng = chnge.ptr<schar>(r);
  //
  //   for (int c=0;c<chnge.cols;++c){
  //     if(*ac > - 10){
  //       if(*ac++ >= *nh++){
  //         *ng++ = rate;
  //       }else{
  //         *ng++ = -rate;
  //       }
  //     }
  //
  //
  //
  //   }
  // }

  }

void psy_reagent(Mat src,Mat activ,Mat inhib,int p,int n,int thr){

  // Mat padded_Mat = src.clone();
  // int pad = n/2;
  // copyMakeBorder( src, padded_Mat, pad, pad, pad, pad, BORDER_REPLICATE);
  //
  // int w_p = p/2;
  // int w_n = n/2;
  // int activ_conc,inhib_conc;
  // // pixVal,
  //
  // for (int row=pad;row<src.rows+pad;row++){
  //
  //   schar* ac = activ.ptr<schar>(row+1-pad);
  //   schar* nh = inhib.ptr<schar>(row+1-pad);
  //
  //   for(int col=pad;col<src.cols+pad;col++){
  //
  //
  //     activ_conc = mean(padded_Mat(Rect(col-w_p,row-w_p,p,p)) )[0];
  //     inhib_conc = mean(padded_Mat(Rect(col-w_n,row-w_n,n,n)) )[0];
  //
  //     if (activ_conc < thr){
  //       activ_conc = rand()%2;
  //       inhib_conc = rand()%2;
  //     }
  //
  //
  //     *ac++ = activ_conc;
  //     *nh++ = inhib_conc;
  //
  //   }
  // }
}

void make_sym(Mat scale){
  Mat clone = scale.clone();
  for(int i=0;i<clone.rows/2+1;i++){
    for(int j=0;j<clone.cols/2+1;j++){
      int a = scale.at<schar>(i,j);
      int b = scale.at<schar>(scale.rows-i,j);
      int c = scale.at<schar>(i,scale.cols-j);
      int d =  scale.at<schar>(scale.rows-i,scale.cols-j);
      double avg = 0.25*(a+b+c+d);
      scale.at<schar>(i,j) = avg;
      scale.at<schar>(scale.rows-i,j) = avg;
      scale.at<schar>(i,scale.cols-j) = avg;
      scale.at<schar>(scale.rows-i,scale.cols-j) = avg;
    }
  }


  // Mat clone = scale.clone();
  // int nRow = clone.rows;
  // int nCols = clone.cols;
  //
  // for(int i=0;i<nRow/2+1;i++){
  //   schar* pix_r = scale.ptr<schar>(i+1);
  //   schar* pix_rp = scale.ptr<schar>(nRow-i-1);
  //
  //
  //   for(int j=0;j<nCols/2+1;j++){
  //
  //     int avg = 0.25*(*(pix_r+j) + *(pix_r+(nCols-j)) + *(pix_rp+j) + *(pix_rp+(nCols-j))  );
  //     *(pix_r+j) = avg;
  //     *(pix_r+(nCols-j)) =avg;
  //     *(pix_rp+j) = avg;
  //     *(pix_rp+(nCols-j)) = avg;
  //   }
  // }

}


double my_rand(){
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);
  return dis(gen);
}


void add_noise(Mat orig,int amount,double prob){
  if (my_rand() >= prob){
    Mat noise = orig.clone();
    randu(noise,Scalar(-amount),Scalar(amount));
    addWeighted(orig, 1, noise, 1,0,orig);
  }

}
