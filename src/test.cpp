#include <iostream>
#include <string>
#include "pointcloud_compressor.h"

using namespace std;

int main(int argc, char** argv)
{
    pointcloud_compressor comp("../data/office2.pcd", 0.08f, 5, 100); // 0.05f, 10
    return 0;
}
