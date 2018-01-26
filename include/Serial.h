#include <iostream>
#include <stdio.h>
#include <time.h>

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#define MANIFOLD 1
#define PC 0
//#if defined (__amd64__) || ( __amd64) ||(__x86_64__) || (__x86_64) ||(i386) ||(__i386) ||(__i386__)
#if defined __arm__
#define PLATFORM MANIFOLD
#else
#define PLATFORM PC
#endif
using namespace std;
class Serial {
private:
#if PLATFORM == MANIFOLD
    int fd;
#endif

private:
    int set_opt(int, int, int, char, int);

public:
    void init();
    void sendTarget(int, int, int);
};
