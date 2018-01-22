#include "GlobalCamera.h"
#include "Serial.h"
//#include <omp.h>
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <stdlib.h>
#include <sys/time.h>

#define VIDEO_FILE 0
#define VIDEO_CAMERA 1
#define VIDEO VIDEO_FILE

#define PI 3.14159265358979323

//#define DRAW 1

using namespace cv;
using namespace std;

double tic()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return ((float)t.tv_sec + ((float)t.tv_usec) / 1000000.);
}

class Armor {
private:
    enum State {
        EXPLORE,
        TRACK_INIT,
        TRACK
    } state;
    Rect2d bbox;
    Serial serial;
    double timer;
    Ptr<Tracker> tracker;
    long found_ctr;
    long unfound_ctr;

    int srcW, srcH;
    int BORDER_IGNORE;
    int BOX_EXTRA;

public:
    void init()
    {
        // init serial
        serial.init();
        // fps
        timer = tic();
        // state machine
        state = EXPLORE;
        found_ctr = 0;
        unfound_ctr = 0;
        srcW = 640;
        srcH = 480;
        BORDER_IGNORE = 10;
        BOX_EXTRA = 10;
    }

    int run(Mat& frame)
    {
        if (frame.empty())
            return -1;
#if VIDEO == VIDEO_FILE
        cvtColor(frame, frame, CV_BGR2GRAY);
#endif
#ifdef DRAW
        imshow("frame", frame);
#endif

        //cout << dec << "found_ctr: " << found_ctr << endl;
        if (state == EXPLORE) {
            if (explore(frame)) {
                ++found_ctr;
                unfound_ctr = 0;
            } else {
                ++unfound_ctr;
                found_ctr = 0;
            }

            if (found_ctr > 1) {
                serial.sendTarget((bbox.x + bbox.width / 2) * 1, (bbox.y + bbox.height / 2) * 1, true);
                transferState(TRACK_INIT);
                found_ctr = 0;
            }
            if (unfound_ctr > 5) {
                serial.sendTarget(320, 240, false);
                unfound_ctr = 0;
            }
        } else if (state == TRACK_INIT) {
            trackInit(frame);
            transferState(TRACK);
        } else if (state == TRACK) {
            if (track(frame)) {
                serial.sendTarget((bbox.x + bbox.width / 2) * 1, (bbox.y + bbox.height / 2) * 1, true);
                ++found_ctr;
            } else {
                transferState(EXPLORE);
                found_ctr = 0;
            }
            if (found_ctr > 100) {
                Mat roi = frame(bbox);
                float mean_roi = mean(roi)[0];
                cout << endl << "mean: " << mean_roi << endl;
                if (mean_roi < 20) {
                    transferState(EXPLORE);
                }
                found_ctr = 0;
            }
        }

#ifdef DRAW
        waitKey(1);
#endif
        return 0;
    }

private:
    void transferState(State s)
    {
        state = s;
    }

    bool explore(Mat& frame)
    {
        static Mat bin;
        threshold(frame, bin, 230, 255, THRESH_BINARY);
#ifdef DRAW
        imshow("gray", bin);
#endif
        vector<vector<Point> > contours;
        vector<RotatedRect> lights;
        findContours(bin, contours,
            CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        for (unsigned int i = 0; i < contours.size(); ++i) {
            int area = contourArea(contours.at(i));
            //cout << "area:" << area << endl;
            if (area > 400 || area < 7) {
#ifdef DRAW
                drawContours(bin, contours, i, Scalar(50), CV_FILLED);
#endif
                continue;
            }
            RotatedRect rec = minAreaRect(contours.at(i));
            Size2f size = rec.size;
            float a = size.height > size.width
                ? size.height
                : size.width;
            float b = size.height < size.width
                ? size.height
                : size.width;
            if (a < 8) {
                continue;
            }
            //cout << "a / b: " << a / b << endl;
            if (a / b > 10 || a / b < 2.5) {
#ifdef DRAW
                drawContours(bin, contours, i, Scalar(100), CV_FILLED);
#endif
                continue;
            }
            lights.push_back(rec);
        }
        if (lights.size() < 2)
            return false;
        int light1 = -1, light2 = -1;
        float min_angel = 3.001;
        for (unsigned int i = 0; i < lights.size(); ++i) {
            for (unsigned int j = i + 1; j < lights.size(); ++j) {
                Point2f pi = lights.at(i).center;
                Point2f pj = lights.at(j).center;
                //float midx = (pi.x + pj.x) / 2;
                //float midy = (pi.y + pj.y) / 2;
                Size2f sizei = lights.at(i).size;
                Size2f sizej = lights.at(j).size;
                float ai = sizei.height > sizei.width
                    ? sizei.height
                    : sizei.width;
                float aj = sizej.height > sizej.width
                    ? sizej.height
                    : sizej.width;
                if (ai / aj > 1.3 || ai / aj < 0.7)
                    continue;

                //灯条中点连线与灯条夹角合适
                float angeli = lights.at(i).angle;
                float angelj = lights.at(j).angle;
                if (sizei.width < sizei.height)
                    angeli += 90.0;
                if (sizej.width < sizej.height)
                    angelj += 90.0;
                //if (abs(angeli - angelj) > 5) {
                //continue;
                //}
                //cout << "i: " << angeli << " j: " << angelj << endl;
                if (abs(angeli - angelj) < min_angel) {
                    float distance_n = abs((pi.x - pj.x) * cos((angeli + 90) * PI / 180)
                        + (pi.y - pj.y) * sin((angeli + 90) * PI / 180));
                    //灯条距离合适
                    //cout << "distance: " << distance
                    //<< " ai: " << ai
                    //<< " aj: " << aj << endl;
                    if (distance_n < 2 * ai || distance_n > 4.5 * ai
                        || distance_n < 2 * aj || distance_n > 4.5 * aj) {
#ifdef DRAW
                        drawContours(bin, contours, i, Scalar(150), CV_FILLED);
                        drawContours(bin, contours, j, Scalar(150), CV_FILLED);
#endif
                        continue;
                    }
                    float distance_t = abs((pi.x - pj.x) * cos((angeli) * PI / 180)
                        + (pi.y - pj.y) * sin((angeli) * PI / 180));
                    //灯条距离合适
                    //cout << "distance: " << distance
                    //<< " ai: " << ai
                    //<< " aj: " << aj << endl;
                    if (distance_t > 1.5 * ai || distance_t > 1.5 * aj) {
#ifdef DRAW
                        drawContours(bin, contours, i, Scalar(150), CV_FILLED);
                        drawContours(bin, contours, j, Scalar(150), CV_FILLED);
#endif
                        continue;
                    }
                    light1 = i;
                    light2 = j;
                    min_angel = abs(angeli - angelj);
                }
            }
        }
        if (light1 == -1 || light2 == -1 || min_angel == 5.001)
            return false;
        //cout << "min i:" << light1 << " j:" << light2 << " angel:" << min_angel << endl;
        Rect2d reci = lights.at(light1).boundingRect();
        Rect2d recj = lights.at(light2).boundingRect();
        float min_x, min_y, max_x, max_y;
        if (reci.x < recj.x) {
            min_x = reci.x;
            max_x = recj.x + recj.width;
        } else {
            min_x = recj.x;
            max_x = reci.x + reci.width;
        }
        if (reci.y < recj.y) {
            min_y = reci.y;
            max_y = recj.y + recj.height;
        } else {
            min_y = recj.y;
            max_y = reci.y + reci.height;
        }
        min_x -= BOX_EXTRA;
        max_x += BOX_EXTRA;
        min_y -= BOX_EXTRA;
        max_y += BOX_EXTRA;
        if (min_x < 0 || max_x > srcW || min_y < 0 || max_y > srcH) {
            return false;
        }
        bbox = Rect2d(min_x, min_y,
            max_x - min_x, max_y - min_y);
#ifdef DRAW
        rectangle(bin, bbox, Scalar(255), 3);
        imshow("gray", bin);
        int k = waitKey(1);
        if (k == 27)
            waitKey(0);
#endif
        return true;
    }

    void trackInit(Mat& frame)
    {
        tracker = TrackerKCF::create();
#ifdef DRAW
        // Display bounding box.
        rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
        imshow("Tracking", frame);
#endif
        tracker->init(frame, bbox);
    }

    bool track(Mat& frame)
    {
        // Update the tracking result
        // threshold(frame, frame, 230, 255, THRESH_BINARY);
        bool ok = tracker->update(frame, bbox);
        if (bbox.x < 10 || bbox.y < 10
                || bbox.x + bbox.width > 630
                || bbox.y + bbox.height > 470) {
            ok = false;
        }

        if (ok) {
#ifdef DRAW
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
#endif

            // send
            float fps = 1 / (tic() - timer);
            cout << "fps: " << fps << endl;
            timer = tic();
#ifdef DRAW
            // Display frame.
            imshow("Tracking", frame);
            int k = waitKey(1);
            if (k == 27)
                waitKey(0);
#endif
            return true;
        } else {
            return false;
        }
    }
};

int main(int argc, char** argv)
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

    Mat frame;

    while (1) {
        video.read(frame);
        if (armor.run(frame) < 0) {
            cout << "Error!" << endl;
            return -1;
        }
    }
}
