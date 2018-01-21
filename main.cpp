#include "GlobalCamera.h"
#include "Serial.h"
#include <omp.h>
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <stdlib.h>
#include <sys/time.h>

#define VIDEO_FILE 0
#define VIDEO_CAMERA 1
#define VIDEO VIDEO_FILE

#define EXPLORE 0
#define TRACK_INIT 1
#define TRACK 2

#define PI 3.14159265358979323

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
    int state;
    Rect2d bbox;
    Serial serial;
    double timer;
    Ptr<Tracker> tracker;

public:
    void init()
    {

        // Get openmp cores
        cout << "available cores:" << omp_get_num_procs() << endl;

        // init serial
        serial.init();

        // fps
        timer = tic();

        // state machine
        state = EXPLORE;
    }

    int run(Mat& frame)
    {
        //video.read(frame);
        if (frame.empty())
            return -1;
#if VIDEO == VIDEO_FILE
        cvtColor(frame, frame, CV_BGR2GRAY);
#endif
        imshow("frame", frame);
        waitKey(1);
        if (state == EXPLORE) {
            explore(frame);
        }
        if (state == TRACK_INIT) {
            trackInit(frame);
        }
        if (state == TRACK) {
            track(frame);
        }
        return 0;
    }

    void explore(Mat& frame)
    {
        static Mat bin;
        threshold(frame, bin, 230, 255, THRESH_BINARY);
        imshow("gray", bin);
        vector<vector<Point> > contours;
        vector<RotatedRect> lights;
        findContours(bin, contours,
            CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        for (unsigned int i = 0; i < contours.size(); ++i) {
            int area = contourArea(contours.at(i));
            //cout << "area:" << area << endl;
            if (area > 400 || area < 30) {
                drawContours(bin, contours, i, Scalar(50), CV_FILLED);
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
            if (a < 10) {
                continue;
            }
            //cout << "a / b: " << a / b << endl;
            if (a / b > 10 || a / b < 2.5) {
                drawContours(bin, contours, i, Scalar(100), CV_FILLED);
                continue;
            }
            lights.push_back(rec);
        }
        if (lights.size() < 2)
            return;
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
                cout << "i: " << angeli << " j: " << angelj << endl;
                if (abs(angeli - angelj) < min_angel) {
                    float distance = abs((pi.x - pj.x) * cos((angeli + 90) * PI / 180)
                            + (pi.y - pj.y) * sin((angeli + 90) * PI / 180));
                    //灯条距离合适
                    cout << "distance: " << distance
                        << " ai: " << ai
                        << " aj: " << aj << endl;
                    if (distance < 2 * ai || distance > 4.5 * ai
                            || distance < 2 * aj || distance > 4.5 * aj) {
                        drawContours(bin, contours, i, Scalar(150), CV_FILLED);
                        drawContours(bin, contours, j, Scalar(150), CV_FILLED);
                        continue;
                    }
                    light1 = i;
                    light2 = j;
                    min_angel = abs(angeli - angelj);
                }
            }
        }
        if (light1 == -1 || light2 == -1 || min_angel == 5.001)
            return;
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
        min_x -= 5;
        max_x += 5;
        min_y -= 5;
        max_y += 5;
        if (min_x < 0)
            min_x = 1;
        if (max_x > frame.size().width)
            max_x = frame.size().width - 1;
        if (min_y < 0)
            min_y = 1;
        if (max_y > frame.size().height)
            max_y = frame.size().height - 1;
        bbox = Rect2d(min_x, min_y,
                max_x - min_x, max_y - min_y);
        rectangle(bin, bbox, Scalar(255), 3);
        //state = TRACK_INIT;
        imshow("gray", bin);
        int k = waitKey(1);
        if (k == 27)
            waitKey(0);
    }

    void trackInit(Mat& frame)
    {
        tracker = TrackerKCF::create();

        // Display bounding box.
        rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
        imshow("Tracking", frame);
        tracker->init(frame, bbox);
        state = TRACK;
    }

    void track(Mat& frame)
    {
        // Update the tracking result
        bool ok = tracker->update(frame, bbox);

        if (ok) {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);

            // send
            serial.sendTarget((bbox.x + bbox.width / 2) * 2, (bbox.y + bbox.height / 2) * 2, ok);
            float fps = 1 / (tic() - timer);
            cout << "fps: " << fps << endl;
            timer = tic();
        } else {
            // Tracking failure detected.
            serial.sendTarget(320, 240, ok);
            state = EXPLORE;
            return;
        }

        // Display frame.
        imshow("Tracking", frame);

        // Exit if ESC pressed.
        int k = waitKey(1);
        if (k == 27) {
            return;
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
