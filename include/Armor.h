#include "Serial.h"
#include "kcftracker.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <sys/time.h>

#define PI 3.14159265358979323
#if PLATFORM == PC
#define DRAW 1
#endif

using namespace cv;
using namespace std;

class Armor {
private:
    enum State {
        EXPLORE,
        TRACK_INIT,
        TRACK
    } state;
    Rect2d bbox;
    Rect2d bbox_last;
    KCFTracker tracker;
    Serial serial;
    double timer;
    long found_ctr;
    long unfound_ctr;

    int srcW, srcH;
    int BORDER_IGNORE;
    int BOX_EXTRA;

    long total_contour_area;

public:
    Armor();
    void init();
    int run(Mat& frame);

private:
    void transferState(State s);
    bool explore(Mat& frame);
    void trackInit(Mat& frame);
    bool track(Mat& frame);
};
