#include "GlobalCamera.h"
#include "Armor.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define VIDEO_FILE 0
#define VIDEO_CAMERA 1

#if PLATFORM == PC
#define VIDEO VIDEO_FILE
#elif PLATFORM == MANIFOLD
#define VIDEO VIDEO_CAMERA
#define OPENMP_SWITCH 1
#endif

using namespace cv;
using namespace std;

int main(void)
{
#if VIDEO == VIDEO_CAMERA
    GlobalCamera video;
#endif
#if VIDEO == VIDEO_FILE
    VideoCapture video;
#endif
    // Read video
#if VIDEO == VIDEO_CAMERA
    if (video.init() == 0) {
        cout << "Global Shutter Camera Init successfully!" << endl;
    } else {
        cout << "Global Shutter Camera Init Failed!" << endl;
        return -1;
    }
#endif
#if VIDEO == VIDEO_FILE
    video.open("/home/jachinshen/视频/Global1.avi");
    if (video.isOpened())
        cout << "Open Video Successfully!" << endl;
    else {
        cout << "Open Video failed!" << endl;
        return -1;
    }
#endif
    Armor armor;
    armor.init();

#ifdef OPENMP_SWITCH
    Mat frame1, frame2;
    bool ok = true;

    video.read(frame2);
    while (ok) {
#pragma omp parallel sections num_threads(2)
        {
#pragma omp section
            {
                video.read(frame1);
            }
#pragma omp section
            {
                if (armor.run(frame2) < 0) {
                    cout << "Error!" << endl;
                    ok = false;
                }
            }
        }

#pragma omp barrier

#pragma omp parallel sections num_threads(2)
        {
#pragma omp section
            {
                video.read(frame2);
            }
#pragma omp section
            {
                if (armor.run(frame1) < 0) {
                    cout << "Error!" << endl;
                    ok = false;
                }
            }
        }
#pragma omp barrier
    }

#else
    Mat frame;
    while(video.read(frame)) {
        armor.run(frame);
    }
    cout << "End!" << endl;
#endif
}
