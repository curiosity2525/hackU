//画像解析、スピーカー、モーター
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv/cv.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <opencv2/objdetect.hpp>

/*音楽再生のsystem関数を利用するためにcstdlibをインクルード*/
#include <cstdlib>

/*シリアル通信に使用*/
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <stdio.h>
#include <strings.h>
#include <signal.h>

using namespace cv;
using namespace std;

namespace cv{
    using std::vector;
    using std::endl;
    using std::cout;
}

/*シリアル通信*/
#define DEV_NAME "/##/######" #デバイス名
#define BAUD_RATE B9600
#define BUFF_SIZE 4096

int fd;
int ret;//復帰値の確認
int tcd = 1;
time_t start_time;


//int fd;
//struct termios oldtio, newtio;
//int isrunning = TRUE;
//ssize_t ret;
//time_t start_time;
//int num;
//int tcd;

//顔検出できた生徒の数を入れる配列
int student_num_data[10];
//データの数
int student_count = 0;
//生徒の数が求められていない時0 求められている時1
int student = 0;
//得られた全ての合計
int all_student_num = 0;
//最終的に求めたい生徒の数
int student_num = 0;

//最大100人の生徒がいるとする
int student_face[100] = {};//全員0

int face_x[100];
int face_y[100];
Mat s_roi[10];
Rect roi[10];

//グレースケール画像、二値化画像
Mat s_roi_gray[100];
Mat s_bin_img[100];

//hsv画像
Mat s_hsv_img[100];

//寝ていると判定されたフレームの数
int sleep_cnt[100];


/*シリアル通信セットアップ*/
void serial_init(int fd) {
    struct termios tio;
    memset(&tio, 0, sizeof(tio));
    tio.c_cflag = CS8 | CLOCAL | CREAD;
    tio.c_cc[VTIME] = 100;
    // ボーレートの設定
    cfsetispeed(&tio, BAUD_RATE);
    cfsetospeed(&tio, BAUD_RATE);
    // デバイスに設定を行う
    tcsetattr(fd, TCSANOW, &tio);
}


/*シリアル通信終了*/
//void serialClose() {
//    tcsetattr(fd,TCSANOW,&oldtio); //restore previous port setting
//    close(fd);
//}

//void interrupt(int sig) {
//    printf("interruput = false");
//    isrunning = FALSE;
//} //ctrl-c handler


//生徒の数は最頻値
int count_student(int student_data[], int student_cnt){
    
    //cnt->出現回数 mode->最頻値
    int cnt = 0, mode = 0, temp_cnt, i, j;
    //int mode_num[student_cnt];
    
    for(i=0; i<student_cnt; i++){
        temp_cnt = 1; //仮の出現回数は1
        for(j = i+1; j < student_cnt; j++){
            if(student_data[i] == student_data[j]){
                temp_cnt++;
            }
        }
        if(temp_cnt > cnt){
            cnt = temp_cnt;
            mode = student_data[i];
            printf("student_data[i] = %d\n", student_data[i]);
        }
    }
    //printf("最頻値は%d", mode);
    student = 2;
    return mode;
}

bool IsSimilar(int ref, int target, int thr){
    if(abs(ref-target)<thr)return 1;
    else if(abs(ref-target+180)<thr||abs(ref-target-180)<thr)return 1;
    else return 0;
}


Mat detectHuman(Mat &image, string &cascade_file, int video_w, int video_h){
    CascadeClassifier cascade;
    cascade.load(cascade_file);
    
    vector<Rect> human;
    cascade.detectMultiScale(image, human, 1.1, 3, 0, Size(20, 20));
    
    if(student == 1){
        student_num = count_student(student_num_data, student_count);
        student_count = 0;
        printf("%d", student_num);
    }
    else if(student == 0){
        student_num_data[student_count] = human.size();
        student_count++;
    }
    //生徒のカウントが終わったら
    else if(student == 2){
        for (int i = 0; i < student_num; i++){
            face_x[i] = (int)human[i].x;
            face_y[i] = (int)human[i].y;
            printf("face[%d]_x = %d\n",i, face_x[i]);
            printf("face[%d]_y = %d\n",i, face_y[i]);
        }
        
        student = 3;
    }
    
    else if(student == 3){
        for(int i=0; i<student_num; i++){
            //Mat clone_img = image.clone();
            //            Rect roi(face_x[i], face_y[i], 100, 100);
            
            //↓はfaces[i].width, faces[i].yにして下のパーセンテージを出すところもs_roi[i].cols, s_roi[i].rowsを使えるようにしたい
            Rect roi(face_x[i], face_y[i], 200, 200);
            s_roi[i] = image(roi);
            //グレースケール、二値化に変更
            cvtColor(s_roi[i], s_roi_gray[i], CV_BGR2GRAY);
            threshold(s_roi_gray[i], s_bin_img[i], 0, 255, THRESH_BINARY|THRESH_OTSU);
            
            cvtColor(s_roi[i], s_hsv_img[i], CV_BGR2HSV);
            
            imshow("a" + i, s_roi[i]);
            //imshow("a" + i, s_bin_img[i]);
            imshow("a" + i, s_hsv_img[i]);
        }
        
        int human_pixels[100] = {};
        //肌色の割合をとる
        for(int i=0; i<student_num; i++){
            for(int y = 0; y < s_roi[i].rows; ++y){
                for(int x = 0; x < s_roi[i].cols; ++x){
                    //肌色である範囲　hsv画像から
                    if(
                       s_hsv_img[i].data[ y * s_hsv_img[i].step + x * s_hsv_img[i].elemSize() + 1 ]>100 &&
                       IsSimilar(s_hsv_img[i].data[ y * s_hsv_img[i].step + x * s_hsv_img[i].elemSize() + 0 ], 15, 10)
                       )
                    {
                        human_pixels[i]++;
                        // printf("HUMAN");
                    }
                }
            }
            // cout<<human_pixels<<" pixels; "<<(int)human_pixels/s_roi[i].cols/s_roi[i].rows*100<<" %"<<endl;
            
            //↓何故かうまくいかない
            /*
             printf("human_pixels[%d] = %d\n", i, human_pixels[i]);
             printf("s_roi[%d].cols = %d\n", i, s_roi[i].cols);
             printf("s_roi[%d].rows = %d\n", i, s_roi[i].rows);
             double human_pix_per = human_pixels[i]/s_roi[i].cols/s_roi[i].rows * 100;
             printf("human[%d] = %lf パーセンテージ\n", i, human_pix_per);
             
             */
            //切り取る画像の大きさによって/100を変えないといけない
            //printf("human[%d] = %d パーセンテージ\n", i,(int)(human_pixels[i]/100));
            
            int human_pct = (int)(human_pixels[i]/400);
            printf("human[%d] = %d パーセンテージ\n", i,human_pct);
            
            if(human_pct < 10){
                sleep_cnt[i]++;
                /*スピーカーで音を鳴らす*/
                if(sleep_cnt[i] > 10){
                    printf("%d is sleeping\n", i);
                    
                    /*シリアル通信開始*/
                    //if(serialOpen() != 0) return -1;
                    //serialOpen();
                    //signal(SIGINT, interrupt);
                    int num;
                    if(face_x[i] < video_w/3){
                        num = 1;
                    }else if(face_x[i] >= video_w/3 && face_x[i] <= video_w*2/3){
                        num = 2;
                    }else{
                        num = 3;
                    }
                    
                    sleep(2);
                    ret = write(fd, &num, 1);
                    tcd = tcdrain( fd );
                    //sleep(2);
                    //復帰値の確認
                    printf("ret = %zd\n", ret);
                    //tcdrainの確認
                    printf("tcd = %d\n", tcd);
                    //serialClose();
                    
                    //system("afplay game_maoudamashii_8_orgel10.mp3");
                    system("音楽ファイル"); //音楽ファイルのパス
                    //student = 4;
                    sleep_cnt[i]=0;
                }
            }else{
                sleep_cnt[i]=0;
            }
        }
        //student = 4;
    }
    
    //10個のデータの合計とする データを10個集める
    if(student_count > 9){
        student = 1;
    }
    
    string window_name[10][20];
    
    return image;
    
}

int main(){
    
    //デバイスファイル(シリアルポート)オープン
    fd = open(DEV_NAME, O_RDWR | O_NONBLOCK);
    if(fd<0) { // デバイスオープンに失敗
        printf("ERROR on device open\n");
        exit(1);
    }
    serial_init(fd); // シリアルポートの初期化
    
    Mat img;
    //動画を開く
    //VideoCapture cap(""); #動画ファイルのパス
    VideoCapture cap(0);
    
    int v_w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int v_h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    
    //動画が開けたか確認
    if(!cap.isOpened()) return -1;
    
    double startTime, endTime;
    double totalTime = 0.0, setTime = 10.0;
    
    while(1){
        //10秒のカウント
        if(totalTime > setTime) {
            //printf("count finish");
            //student = 1;
        }
        endTime = clock() / CLOCKS_PER_SEC ;
        totalTime = endTime - startTime;
        
        Mat frame;
        cap>>frame;
        
        //顔
        string filename("/usr/local/Cellar/opencv/3.4.2/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml");
        Mat detectHumanImage = detectHuman(frame, filename, v_w, v_h);
        Mat gray_img;
        //resize(frame, frame, Size(), 0.5, 0.5);
        imshow("in", frame);
        
        waitKey(1);
    }
}
