#define MANIFOLD 1
#define PC 0

#define VIDEO_FILE 0
#define VIDEO_CAMERA 1

#define PI 3.14159265358979323

#if defined __arm__
#define PLATFORM MANIFOLD
#else
#define PLATFORM PC
#endif

#if PLATFORM == PC

#define VIDEO VIDEO_CAMERA
#define DRAW 1

#elif PLATFORM == MANIFOLD

#define VIDEO VIDEO_CAMERA
#define OPENMP_SWITCH 1
#endif
