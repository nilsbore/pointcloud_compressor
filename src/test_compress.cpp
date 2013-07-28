#include <iostream>
#include <string>
#include "pointcloud_compressor.h"

using namespace std;

int main(int argc, char** argv)
{
    pointcloud_compressor comp("../data/office1.pcd", 0.2f, 20, 200, 10,
                               1e-2f, 1e-4f, 300, 20, 5e4f, 1e2f);
    comp.save_compressed("test");
    return 0;
}