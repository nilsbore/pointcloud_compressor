#include <iostream>
#include <string>
#include "pointcloud_compressor.h"

using namespace std;

int main(int argc, char** argv)
{
    pointcloud_compressor comp("../data/office1.pcd", 0.1f, 6, 100, 30, 1e-5f, 1e-6f); // 0.05f, 10
    return 0;
}
