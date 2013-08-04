#include <iostream>
#include <string>
#include "pointcloud_compressor.h"

using namespace std;

int main(int argc, char** argv)
{
    pointcloud_compressor comp("../data/office1.pcd", 0.2f, 20, 50, 10,
                               5e-2f, 1e-3f, 100, 20, 1e5f, 1e3f);
    comp.save_compressed("test");

    return 0;
}
