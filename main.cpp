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
#define TRACK 1

#define PI 3.14159265358979323

using namespace cv;
using namespace std;

int state;
Rect2d bbox;

double tic()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return ((float)t.tv_sec + ((float)t.tv_usec) / 1000000.);
}

void explore(Mat& frame)
{
    static Mat bin;
    threshold(frame, bin, 230, 255, THRESH_BINARY);
    vector<vector<Point> > contours;
    vector<RotatedRect> lights;
    findContours(bin, contours,
        CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    for (unsigned int i = 0; i < contours.size(); ++i) {
        int area = contourArea(contours.at(i));
        //cout << "area:" << area << endl;
        if (area > 400 || area < 100) {
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
        if (a / b > 4.5 || a / b < 2) {
            drawContours(bin, contours, i, Scalar(100), CV_FILLED);
            continue;
        }
        lights.push_back(rec);
    }
    if (lights.size() < 2)
        return;
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
            float distance = sqrt((pi.x - pj.x) * (pi.x - pj.x)
                + (pi.y - pj.y) * (pi.y - pj.y));
            //灯条距离合适
            //cout << "distance: " << distance
            //<< " ai: " << ai
            //<< " aj: " << aj << endl;
            if (distance < 1 * ai || distance > 5 * ai
                || distance < 1 * aj || distance > 5 * aj) {
                drawContours(bin, contours, i, Scalar(150), CV_FILLED);
                drawContours(bin, contours, j, Scalar(150), CV_FILLED);
                continue;
            }

            //灯条中点连线与灯条夹角合适
            float angeli = lights.at(i).angle;
            float angelj = lights.at(j).angle;
            if (sizei.width < sizei.height)
                angeli += 90;
            if (sizej.width < sizej.height)
                angelj += 90;
            if (abs(angeli - angelj) > 5) {
                continue;
            }
            //cout << "i: " << angeli << " j: " << angelj << endl;
            Rect2d reci = lights.at(i).boundingRect();
            Rect2d recj = lights.at(j).boundingRect();
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
            bbox = Rect2d(min_x, min_y,
                max_x - min_x, max_y - min_y);
            rectangle(bin, bbox, Scalar(255), 3);
            state = TRACK;
        }
    }
    imshow("gray", bin);
    waitKey(1);
}

// Convert to string
#define SSTR(x) static_cast<std::ostringstream&>( \
    (std::ostringstream() << std::dec << x))      \
                    .str()

int main(int argc, char** argv)
{
// Read video
#if VIDEO == VIDEO_CAMERA
    GlobalCamera video;
    if (video.init() == 0) {
        cout << "Global Shutter Camera Init successfully!" << endl;
    } else {
        cout << "Global Shutter Camera Init Failed!" << endl;
        return -1;
    }
#endif
#if VIDEO == VIDEO_FILE
    VideoCapture video("/home/jachinshen/视频/Global1.avi");
#endif

    // Get openmp cores
    cout << "available cores:" << omp_get_num_procs() << endl;

    // init serial
    Serial serial;
    serial.init();

    // fps
    double timer = tic();

    // state machine
    state = EXPLORE;

    Mat frame;
    // Define initial boundibg box

    while (1) {
        // Read first frame
        video.read(frame);
#if VIDEO == VIDEO_FILE
        cvtColor(frame, frame, CV_BGR2GRAY);
#endif
        imshow("frame", frame);
        waitKey(1);
        if (state == EXPLORE) {
            explore(frame);
        }

        while (state == TRACK) {

            Ptr<Tracker> tracker;

            tracker = TrackerKCF::create();

            // Uncomment the line below to select a different bounding box
            //bbox = selectROI(frame, false);

            // Display bounding box.
            rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
            imshow("Tracking", frame);

            tracker->init(frame, bbox);

            while (1) {

                video.read(frame);

#if VIDEO == VIDEO_FILE
                cvtColor(frame, frame, CV_BGR2GRAY);
#endif

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
                    break;
                }

                // Display frame.
                imshow("Tracking", frame);

                // Exit if ESC pressed.
                int k = waitKey(1);
                if (k == 27) {
                    break;
                }
            }
        }
    }
}
