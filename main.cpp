#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <opencv2/core/ocl.hpp>
#include "GlobalCamera.h"
#include <omp.h>
#include "Serial.h"
#include <stdlib.h>
#include <sys/time.h>

 
using namespace cv;
using namespace std;

double tic()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return ((double)t.tv_sec + ((double)t.tv_usec) / 1000000.);
}
 
// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()
 
int main(int argc, char **argv)
{
    // List of tracker types in OpenCV 3.2
    // NOTE : GOTURN implementation is buggy and does not work.
    string trackerTypes[6] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN"};
    // vector <string> trackerTypes(types, std::end(types));
 
    // Create a tracker
    string trackerType = trackerTypes[2];
 
    // Read video
    GlobalCamera video;
    cout << "init" << video.init() << endl;
    cout << "available cores:" << omp_get_num_procs() << endl;
    Serial serial;
    serial.init();
    double timer = tic();
    //VideoCapture video("../cut2.avi");
    while(1)
    {

    Ptr<Tracker> tracker;
 
    #if (CV_MINOR_VERSION < 3)
    {
        tracker = Tracker::create(trackerType);
    }
    #else
    {
        tracker = TrackerKCF::create();
    }
    #endif
     
     
    // Read first frame
    Mat frame;
    bool ok = video.read(frame);
    //cvtColor(frame, frame, CV_GRAY2BGR);
     
    // Define initial boundibg box
    Rect2d bbox(287, 23, 86, 320);
     
    // Uncomment the line below to select a different bounding box
    bbox = selectROI(frame, false);
 
    // Display bounding box.
    rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
    imshow("Tracking", frame);
     
    tracker->init(frame, bbox);
    
    while(1)
    {     

	
	video.read(frame);
	

	
	
        //cvtColor(frame, frame, CV_GRAY2BGR);
        // Start timer
        //double timer = (double)getTickCount();
	
         
        // Update the tracking result
        bool ok = tracker->update(frame, bbox);
         
        // Calculate Frames per second (FPS)
        //float fps = getTickFrequency() / ((double)getTickCount() - timer);
	
         
        if (ok)
        {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
	    
	    serial.sendTarget((bbox.x + bbox.width / 2) * 2, (bbox.y + bbox.height / 2) * 2, ok);
    	    float fps = 1 / (tic() - timer);
  	    cout << "fps: " << fps << endl;
	    timer = tic();
        }
        else
        {
            // Tracking failure detected.
            //putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
	    serial.sendTarget(320, 240, ok);
        }
         
        // Display tracker type on frame
        //putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
         
        // Display FPS on frame
        //putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
 
        // Display frame.
        imshow("Tracking", frame);

         
        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
 
    }
    }
}
