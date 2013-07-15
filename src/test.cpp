#include <iostream>
#include <string>
#include "pointcloud_compressor.h"

using namespace std;

int main(int argc, char** argv)
{
    pointcloud_compressor comp("../data/office1.pcd", 0.1f, 20); // 0.05f, 10
    return 0;
}
