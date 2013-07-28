#include <iostream>
#include <string>
#include "pointcloud_decompressor.h"

using namespace std;

int main(int argc, char** argv)
{
    pointcloud_decompressor decomp;
    decomp.load_compressed("test");

    return 0;
}
