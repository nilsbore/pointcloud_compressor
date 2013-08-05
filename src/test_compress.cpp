#include <iostream>
#include <string>
#include "pointcloud_compressor.h"

using namespace std;

int main(int argc, char** argv)
{
    pointcloud_compressor comp("/home/nbore/Downloads/home_data_ascii/scene11_ascii.pcd", 0.2f, 30, 200, 10,
                               5e-3f, 1e-4f, 400, 20, 1e4f, 1e2f);
    comp.save_compressed("test");

    return 0;
}
