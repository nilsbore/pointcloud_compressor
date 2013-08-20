#include <iostream>
#include <string>
#include "pointcloud_compressor.h"

using namespace std;

int main(int argc, char** argv)
{
    //pointcloud_compressor comp("/home/nbore/Downloads/home_data_ascii/scene11_ascii.pcd", 0.2f, 40, 400, 10, 5e-2f, 1e-4f, 800, 20, 5e4f, 1e2f);
    pointcloud_compressor comp("/home/nbore/Downloads/home_data_ascii/scene11_ascii.pcd", 0.2f, 20, 200, 10,
                               5e-2f, 1e-4f, 400, 20, 5e4f, 1e2f);
    comp.save_compressed("test");

    return 0;
}
