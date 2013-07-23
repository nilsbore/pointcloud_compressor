#include <iostream>
#include <string>
#include "pointcloud_compressor.h"
#include "pointcloud_decompressor.h"

using namespace std;

int main(int argc, char** argv)
{
    if (false) {
        pointcloud_compressor comp("../data/office1.pcd", 0.1f, 10, 100, 10, 1e-3f, 1e-2f);
        comp.save_compressed("test");
    }
    else {
        pointcloud_decompressor decomp;
        decomp.load_compressed("test");
    }
    return 0;
}
