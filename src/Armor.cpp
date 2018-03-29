#include "Armor.h"
#define _SIZE_ 100000*sizeof(char)
#define _PATH_ "/home/li/test4/fifo"

double tic()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return ((float)t.tv_sec + ((float)t.tv_usec) / 1000000.);
}

Armor::Armor()
    : tracker(false, true, false, false){};

void Armor::init()
{
    // init serial
    //serial.init();

    // fps
    timer = tic();

    // state machine
    state = EXPLORE;

    found_ctr = 0;
    unfound_ctr = 0;

    srcW = 640;
    srcH = 480;

    // track
    BORDER_IGNORE = 10;
    BOX_EXTRA = 10;

    GRAY_THRESH = 235;

    // select contours
    CONTOUR_AREA_MIN = 20;
    CONTOUR_AREA_MAX = 2000;
    CONTOUR_LENGTH_MIN = 20;
    CONTOUR_HW_RATIO_MIN = 2.5;
    CONTOUR_HW_RATIO_MAX = 15;

    // pair lights
    TWIN_ANGEL_MAX = 5.001;
    TWIN_LENGTH_RATIO_MAX = 1.2;
    TWIN_DISTANCE_N_MIN = 1.7;
    TWIN_DISTANCE_N_MAX = 3.8;
    TWIN_DISTANCE_T_MAX = 1.4;

    // state machine
    EXPLORE_TRACK_THRES = 2;
    EXPLORE_SEND_STOP_THRES = 5;
    TRACK_CHECK_THRES = 3;
    TRACK_CHECK_RATIO = 0.4;
    TRACK_EXPLORE_THRES = 2;
}

int Armor::run(Mat& frame)
{
    if (frame.empty())
        return -1;
#if VIDEO == VIDEO_FILE
    cvtColor(frame, frame, CV_BGR2GRAY);
#endif
#ifdef DRAW
    imshow("frame", frame);
#endif

    if (state == EXPLORE) {
        if (explore(frame)) {
            ++found_ctr;
            unfound_ctr = 0;
        } else {
            ++unfound_ctr;
            found_ctr = 0;
        }

        if (found_ctr >= EXPLORE_TRACK_THRES) {
            sendTarget((bbox.x + bbox.width / 2),
                    (bbox.y + bbox.height / 2), FOUND_BORDER);
            transferState(TRACK_INIT);
            found_ctr = 0;
            unfound_ctr = 0;
            bbox_last = bbox;
        }
        if (unfound_ctr >= EXPLORE_SEND_STOP_THRES) {
            sendTarget(srcW / 2, srcH / 2, NOT_FOUND);
            found_ctr = 0;
            unfound_ctr = 0;
        }
    } else if (state == TRACK_INIT) {
        trackInit(frame);
        transferState(TRACK);
    } else if (state == TRACK) {
        if (track(frame)) {
            float fps = 1 / (tic() - timer);
            cout << "fps: " << fps << endl;
            timer = tic();

            int x = bbox.x + bbox.width / 2;
            int y = bbox.y + bbox.height / 2;
            int x_last = bbox_last.x + bbox_last.width / 2;
            //int y_last = bbox_last.y + bbox_last.height / 2;
            int center_x = 2 * x - srcW / 2;
            int center_y = 2 * y - srcH / 2;
            // Assume the box run at const velocity
            // Predict if the center is still in box at next frame
            if (bbox_last.x < center_x
                    && center_x < bbox_last.x + bbox_last.width
                    && bbox_last.y < center_y
                    && center_y < bbox_last.y + bbox_last.height) {
                // if center is in box, predict it run at const velocity
                sendTarget(2 * x - x_last, y, FOUND_CENTER);
            } else {
                sendTarget(x, y, FOUND_BORDER);
            }
            ++found_ctr;
            unfound_ctr = 0;
            bbox_last = bbox;
        } else {
            ++unfound_ctr;
            found_ctr = 0;
        }

        // check if the box is still tracking armor
        if (found_ctr >= TRACK_CHECK_THRES) {
            Mat roi = frame(bbox);
            threshold(roi, roi, GRAY_THRESH, 255, THRESH_BINARY);
            if (countNonZero(roi) < TRACK_CHECK_RATIO * total_contour_area) {
                transferState(EXPLORE);
                found_ctr = 0;
                unfound_ctr = 0;
            }
        }
        // sometimes, tracker only miss 1 frame
        if (unfound_ctr >= TRACK_EXPLORE_THRES) {
            transferState(EXPLORE);
            unfound_ctr = 0;
            found_ctr = 0;
        }
#ifdef DRAW
        // Draw the tracked object
        rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
        // Display frame.
        imshow("Tracking", frame);
        waitKey(1);
#endif
    }
    return 0;
}

void Armor::transferState(State s)
{
    state = s;
}

void Armor::trackInit(Mat& frame)
{
#ifdef DRAW
    // Display bounding box.
    rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
    imshow("TrackInit", frame);
#endif
    tracker.init(bbox, frame);
}

bool Armor::track(Mat& frame)
{
    // Update the tracking result
    bool ok = true;
    bbox = tracker.update(frame);
    if (bbox.x < BORDER_IGNORE || bbox.y < BORDER_IGNORE
        || bbox.x + bbox.width > srcW - BORDER_IGNORE
        || bbox.y + bbox.height > srcH - BORDER_IGNORE) {
        ok = false;
    }
    return ok;
}

bool Armor::explore(Mat& frame)
{
    static Mat bin;
    threshold(frame, bin, GRAY_THRESH, 255, THRESH_BINARY);
#ifdef DRAW
    imshow("gray", bin);
#endif
    vector<vector<Point> > contours;
    vector<RotatedRect> lights;
    vector<long> areas;
    findContours(bin, contours,
        CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    //select contours by area, length, width/height
    for (unsigned int i = 0; i < contours.size(); ++i) {
        long area = contourArea(contours.at(i));
        //cout << "area:" << area << endl;
        if (area > CONTOUR_AREA_MAX
                || area < CONTOUR_AREA_MIN) {
#ifdef DRAW
            drawContours(bin, contours, i, Scalar(50), CV_FILLED);
#endif
            continue;
        }
        RotatedRect rec = minAreaRect(contours.at(i));
        Size2f size = rec.size;
        // get a (longer) as length
        float a = size.height > size.width
            ? size.height
            : size.width;
        float b = size.height < size.width
            ? size.height
            : size.width;
        if (a < CONTOUR_LENGTH_MIN) {
            continue;
        }
        //check if it is thin
        //cout << "a / b: " << a / b << endl;
        if (a / b > CONTOUR_HW_RATIO_MAX
                || a / b < CONTOUR_HW_RATIO_MIN) {
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
    float min_angel = TWIN_ANGEL_MAX;
    // pair lights by length, distance, angel
    for (unsigned int i = 0; i < lights.size(); ++i) {
        for (unsigned int j = i + 1; j < lights.size(); ++j) {
            Point2f pi = lights.at(i).center;
            Point2f pj = lights.at(j).center;
            Size2f sizei = lights.at(i).size;
            Size2f sizej = lights.at(j).size;
            float ai = sizei.height > sizei.width
                ? sizei.height
                : sizei.width;
            float aj = sizej.height > sizej.width
                ? sizej.height
                : sizej.width;
            // length similar
            if (ai / aj > TWIN_LENGTH_RATIO_MAX
                    || aj / ai > TWIN_LENGTH_RATIO_MAX)
                continue;

            // angel parallel
            float angeli = lights.at(i).angle;
            float angelj = lights.at(j).angle;
            if (sizei.width < sizei.height)
                angeli += 90.0;
            if (sizej.width < sizej.height)
                angelj += 90.0;
            if (abs(angeli - angelj) < min_angel) {
                float distance_n = abs((pi.x - pj.x) * cos((angeli + 90) * PI / 180)
                    + (pi.y - pj.y) * sin((angeli + 90) * PI / 180));
                // normal distance range in about 1 ~ 2 times of length
                if (distance_n < TWIN_DISTANCE_N_MIN * ai || distance_n > TWIN_DISTANCE_N_MAX * ai
                    || distance_n < TWIN_DISTANCE_N_MIN * aj || distance_n > TWIN_DISTANCE_N_MAX * aj) {
#ifdef DRAW
                    drawContours(bin, contours, i, Scalar(150), CV_FILLED);
                    drawContours(bin, contours, j, Scalar(150), CV_FILLED);
#endif
                    continue;
                }
                // direction distance should be small
                float distance_t = abs((pi.x - pj.x) * cos((angeli)*PI / 180)
                    + (pi.y - pj.y) * sin((angeli)*PI / 180));
                if (distance_t > TWIN_DISTANCE_T_MAX * ai || distance_t > TWIN_DISTANCE_T_MAX * aj) {
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
    if (light1 == -1 || light2 == -1 || min_angel == TWIN_ANGEL_MAX)
        return false;
    //cout << "min i:" << light1 << " j:" << light2 << " angel:" << min_angel << endl;
    // get and extend box for track init
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

void Armor::sendTarget(int target_x, int target_y, int is_found)
{
    if (target_x < 0 || target_x > 640 || target_y < 0 || target_y > 480) {
        is_found = 0;
        target_x = 320;
        target_y = 240;
    }
    cout << dec << "x:" << target_x << " y:" << target_y << endl;

    char buf[7];
    buf[0] = 0xA5;
    buf[1] = (target_x >> 8) & 0xFF;
    buf[2] = target_x & 0xFF;
    if (is_found == 2) {
        buf[3] = 0xA6;
    } else if (is_found == 1) {
        buf[3] = 0xA6;
    } else {
        buf[3] = 0xA4;
    }
    buf[4] = (target_y >> 8) & 0xFF;
    buf[5] = target_y & 0xFF;
    buf[6] = 0xA7;

    for (int i = 0; i < 7; ++i) {
        cout << hex << (unsigned int)buf[i] << " ";
    }
    cout << endl;
    if((mkfifo(_PATH_, 0777) < 0) && (errno != EEXIST))  
    {  
        printf("cannot create fifo...\n");  
        exit(1);  
    }
    int fd;
    while(( fd = open (_PATH_, O_WRONLY|O_NONBLOCK) )< 0){
        cout << "open error!" << endl;
    

    }
//    for(int i = 0; fd < 0 && i;)

    //char* pipeptr = NULL;

    //memset(pipeptr, '\0', _SIZE_);

    if(write(fd,buf,7) < 0){
        cout << "write error!" << endl;
    }
    close(fd);
}
