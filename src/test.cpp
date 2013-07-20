#include <iostream>
#include <string>
#include "pointcloud_compressor2.h"

using namespace std;

int main(int argc, char** argv)
{
    pointcloud_compressor comp("../data/office1.pcd", 0.1f, 10, 100, 20, 1e-3f, 1e-5f); // 0.05f, 10
    return 0;
}
