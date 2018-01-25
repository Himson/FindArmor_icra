#include "GlobalCamera.h"
#include "Serial.h"
#include <omp.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "kcftracker.hpp"
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
    Armor():
        tracker(false, true, false, false)
    {};

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

            if (found_ctr > 2) {
                serial.sendTarget((bbox.x + bbox.width / 2), (bbox.y + bbox.height / 2), 1);
                transferState(TRACK_INIT);
                found_ctr = 0;
                unfound_ctr = 0;
                bbox_last = bbox;
            }
            if (unfound_ctr > 5) {
                serial.sendTarget(320, 240, 0);
                found_ctr = 0;
                unfound_ctr = 0;
            }
        } else if (state == TRACK_INIT) {
            trackInit(frame);
            transferState(TRACK);
        } else if (state == TRACK) {
            if (track(frame)) 
            {
                float fps = 1 / (tic() - timer);
                cout << "fps: " << fps << endl;
                timer = tic();

                int x = bbox.x + bbox.width / 2;
                int y = bbox.y + bbox.height / 2;
                int x_last = bbox_last.x + bbox_last.width / 2;
                int y_last = bbox_last.y + bbox_last.height / 2;
                int center_x = 2 * x - srcW / 2;
                int center_y = 2 * y - srcH / 2;
                if (bbox_last.x < center_x && center_x < bbox_last.x + bbox_last.width
                        && bbox_last.y < center_y && center_y < bbox_last.y + bbox_last.height) {
                    serial.sendTarget(2 * x - x_last, y, 2);
                } else {
                    serial.sendTarget(x, y, 1);
                }
                ++found_ctr;
                unfound_ctr = 0;
                bbox_last = bbox;
            } else {
                ++unfound_ctr;
                found_ctr = 0;
            }
            if (found_ctr > 3) 
            {
                Mat roi = frame(bbox);
                threshold(roi, roi, 240, 255, THRESH_BINARY);
                if (countNonZero(roi) < 0.4 *  total_contour_area) {
                    transferState(EXPLORE);
                    found_ctr = 0;
                    unfound_ctr = 0;
                }
            }
            if (unfound_ctr > 2) {
                transferState(EXPLORE);
                unfound_ctr = 0;
                found_ctr = 0;
            }
#ifdef DRAW
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
            // Display frame.
            imshow("Tracking", frame);
#endif
        }
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
        threshold(frame, bin, 235, 255, THRESH_BINARY);
#ifdef DRAW
        imshow("gray", bin);
#endif
        vector<vector<Point> > contours;
        vector<RotatedRect> lights;
        vector<long> areas;
        findContours(bin, contours,
            CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        for (unsigned int i = 0; i < contours.size(); ++i) {
            long area = contourArea(contours.at(i));
            //cout << "area:" << area << endl;
            if (area > 2000 || area < 20) {
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
            if (a < 20) {
                continue;
            }
            //cout << "a / b: " << a / b << endl;
            if (a / b > 15 || a / b < 2.5) {
#ifdef DRAW
                drawContours(bin, contours, i, Scalar(100), CV_FILLED);
#endif
                continue;
            }
            lights.push_back(rec);
            areas.push_back(area);
        }
        if (lights.size() < 2)
            return false;
        int light1 = -1, light2 = -1;
        float min_angel = 5.001;
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
                if (ai / aj > 1.2 || aj / ai > 1.2)
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
                    if (distance_n < 1.7 * ai || distance_n > 3.8 * ai
                            || distance_n < 1.7 * aj || distance_n > 3.8 * aj) {
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
                    if (distance_t > 1.4 * ai || distance_t > 1.4 * aj) {
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
        total_contour_area = areas.at(light1) + areas.at(light2);
#ifdef DRAW
        rectangle(bin, bbox, Scalar(255), 3);
        imshow("gray", bin);
#endif
        return true;
    }

    void trackInit(Mat& frame)
    {
#ifdef DRAW
        // Display bounding box.
        rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
        imshow("TrackInit", frame);
#endif
        tracker.init(bbox, frame);
    }

    bool track(Mat& frame)
    {
        // Update the tracking result
        bool ok = true;
        bbox = tracker.update(frame);
        if (bbox.x < 10 || bbox.y < 10
                || bbox.x + bbox.width > 630
                || bbox.y + bbox.height > 470) {
            ok = false;
        }
        return ok;
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

    Mat frame1, frame2, frame;
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
                if (armor.run(frame2) < 0)
                {
                    cout << "Error!" << endl;
                    ok = false;
                }
            }
        }

#pragma omp barrier

#ifdef DRAW
            waitKey(1);
#endif

#pragma omp parallel sections num_threads(2)
        {
#pragma omp section
            {
                video.read(frame2);
            }
#pragma omp section
            {
                if (armor.run(frame1) < 0)
                {
                    cout << "Error!" << endl;
                    ok = false;
                }
            }
        }
#pragma omp barrier
#ifdef DRAW
            waitKey(1);
#endif
    }
}
