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
// void apply_gradient(Mat,Mat);
void addW(Mat,Mat,int);

///////////////////////

int main(){
  //
  srand(time(NULL));
  int seed = 1851156740;//rand();
  theRNG().state = seed;
  cout << "Random Seed: " << seed << endl;
  //
  Size size(400,400);
  Mat frame(size,CV_8SC1);
  randu(frame,Scalar(-127),Scalar(127));
  Mat display,displayRGB;
  //
  //
  int num_scales = 2;
  vector<Mat> activ_vec;
  vector<Mat> inhib_vec;
  for(int v=0;v<num_scales;v++){
    activ_vec.push_back(Mat::zeros(size,CV_8SC1));
    inhib_vec.push_back(Mat::zeros(size,CV_8SC1));
  }
  Mat global_chnge = Mat::zeros(size,CV_8SC1); // how to change
  Mat global_activ = Mat::zeros(size,CV_8SC1); // merger of all
  Mat global_inhib = Mat::zeros(size,CV_8SC1);
  //
  // ONLY ODD Nums & N[*] > P[*] Square kernels/scales
  int P[] = {23,7,27};
  int N[] = {33,17,29};
  int weights[] = {1,4,2};

  int rate = 1;
  //
  int loops = 5000;
  int make_sym_FLAG = 1;


  namedWindow("Frame",WINDOW_AUTOSIZE);
  moveWindow("Frame",1180,464);
  imshow("Frame",frame);
  waitKey(20);


  for(int i=0;i<loops;i++){

    cout << "\r*Iteration: " << i << flush;

    // threads are manually set, more than 4 is not beneficial
    thread t1(calc_reagnet,frame, activ_vec[0],inhib_vec[0], P[0], N[0]);
    thread t2(calc_reagnet,frame, activ_vec[1],inhib_vec[1], P[1], N[1]);
    // thread t3(calc_reagnet,frame, activ_vec[2],inhib_vec[2], P[2], N[2]);
    // thread t4(calc_reagnet,frame, activ_vec[3],inhib_vec[3], P[3], N[3]);

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



    for (int c=0;c<num_scales;c++){

      addWeighted(global_activ, 1, activ_vec[c], weights[c],0,global_activ);
      addWeighted(global_inhib, 1, inhib_vec[c], weights[c],0,global_inhib);
    }


    calc_scale(global_activ,global_inhib,global_chnge,rate);


    addWeighted(frame, 1, global_chnge, 1,0,frame);
    normalize(frame,  frame, -127, 127, NORM_MINMAX);

    imshow("Frame",frame);


    global_chnge = 0;
    global_activ = 0;
    global_inhib = 0;

    if(waitKey(1) >= 0){
      Mat wFrame;
      frame.convertTo(wFrame, CV_8UC1, 255);
      imwrite("./imgs/img_"+to_string(seed)+".png",wFrame);
      cout << endl << "Pattern Saved" << endl;
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

/*
void apply_gradient(Mat src,Mat grad_mat){
  for(int row=0;row<src.rows;row++){
    for(int col=0;col<src.cols;col++){
      src.at<schar>(col,row) += grad_mat.at<schar>(col,row);
    }
  }
}*/

void addW(Mat global,Mat scale, int weight){
  Mat temp = global.clone();
  global  = temp + weight*scale;

  // Too much syntax
  // for(int r=0;r<global.rows;++row){
  //   schar* g_pix = global.ptr<schar>(r);
  //   schar* s_pix = scale.ptr<schar>(r);
  //   for (int c=0;c<global.cols;++c){
  //     *g_pix++ += *s_pix++ * weight;
  //   }
  // }
  //// SLOW !
  // int update,current;
  // for (int row=0;row<global.rows;row++){
  //   for(int col=0;col<global.cols;col++){
  //     update = weight*scale.at<schar>(col,row);
  //     current = global.at<schar>(col,row);
  //     global.at<schar>(col,row) = update + current;
  //   }
  // }


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

      // pixVal = padded_Mat.at<schar>(col,row);

      activ_conc = mean(padded_Mat(Rect(col-w_p,row-w_p,p,p)) )[0];
      inhib_conc = mean(padded_Mat(Rect(col-w_n,row-w_n,n,n)) )[0];

      *ac++ = activ_conc;
      *nh++ = inhib_conc;
      // activ.at<schar>(row-pad,col-pad) = activ_conc;
      // inhib.at<schar>(row-pad,col-pad) = inhib_conc;

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
  // int ac,nh;
  // for (int row=0;row<activ.rows;row++){
  //   for(int col=0;col<activ.cols;col++){
  //     ac = activ.at<schar>(col,row);
  //     nh = inhib.at<schar>(col,row);
  //     if (ac >= nh){
  //       chnge.at<schar>(col,row) = rate;
  //     }else{
  //       chnge.at<schar>(col,row) = -rate;
  //     }
  //
  //     }
  //   }

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
  // --
  // for(int i=0;i<clone.rows/2+1;i++){
  //   for(int j=0;j<clone.cols/2+1;j++){
  //     int avg = 0.25*(clone.at<schar>(j,i) + clone.at<schar>(j,clone.rows-i) + clone.at<schar>(clone.cols-j) + clone.at<schar>(clone.cols-j,clone.rows-i));
  //     scale.at<schar>(j,i) = avg;
  //     scale.at<schar>(j,scale.rows-i) = avg;
  //     scale.at<schar>(scale.cols-j,i) = avg;
  //     scale.at<schar>(scale.cols-j,scale.rows-i) = avg;
  //   }
  // }
  // --
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
