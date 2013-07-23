#include "pointcloud_decompressor.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <stdint.h>
#include <boost/thread/thread.hpp>
#include <fstream>

using namespace Eigen;

pointcloud_decompressor::pointcloud_decompressor()
{

}

void pointcloud_decompressor::load_compressed(const std::string& name)
{
    read_from_file(name);
    decompress_depths();
    decompress_colors();
    reproject_cloud();
}

void pointcloud_decompressor::decompress_depths()
{
    for (int i = 0; i < S.cols(); ++i) {
        S.col(i).setZero();
        for (int k = 0; k < number_words[i]; ++k) {
            S.col(i) += X(k, i)*D.col(I(k, i));
        }
    }
}

void pointcloud_decompressor::decompress_colors()
{
    for (int i = 0; i < RGB.cols(); ++i) {
        RGB.col(i).setZero();
        for (int k = 0; k < RGB_number_words[i]; ++k) {
            RGB.col(i) += RGB_X(k, i)*RGB_D.col(RGB_I(k, i));
        }
    }
}

void pointcloud_decompressor::reproject_cloud()
{
    int n = S.cols();
    pointcloud::Ptr ncloud(new pointcloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ncenters(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ncloud->width = n*sz*sz;
    ncenters->width = n;
    normals->width = n;
    ncloud->height = 1;
    ncenters->height = 1;
    normals->height = 1;
    ncloud->points.resize(ncloud->width * ncloud->height);
    ncenters->points.resize(ncenters->width * ncenters->height);
    normals->points.resize(normals->width * normals->height);
    Vector3f pt;
    int counter = 0;
    int ind;
    for (int i = 0; i < n; ++i) {
        for (int y = 0; y < sz; ++y) { // ROOM FOR SPEEDUP
            for (int x = 0; x < sz; ++x) {
                ind = x*sz + y;
                if (!W(ind, i)) {
                    continue;
                }
                pt(0) = S(ind, i);
                pt(1) = (float(x) + 0.5f)*res/float(sz) - res/2.0f;
                pt(2) = (float(y) + 0.5f)*res/float(sz) - res/2.0f;
                pt = rotations[i]*pt + means[i];
                ncloud->at(counter).x = pt(0);
                ncloud->at(counter).y = pt(1);
                ncloud->at(counter).z = pt(2);
                ncloud->at(counter).r = short(RGB_means[i](0) + RGB(ind, i));
                ncloud->at(counter).g = short(RGB_means[i](1) + RGB(ind, S.cols() + i));
                ncloud->at(counter).b = short(RGB_means[i](2) + RGB(ind, 2*S.cols() + i));
                ++counter;
            }
        }
        ncenters->at(i).x = means[i](0);
        ncenters->at(i).y = means[i](1);
        ncenters->at(i).z = means[i](2);
        normals->at(i).normal_x = rotations[i](0, 0);
        normals->at(i).normal_y = rotations[i](1, 0);
        normals->at(i).normal_z = rotations[i](2, 0);
    }
    ncloud->resize(counter);
    std::cout << "Size of transformed point cloud: " << ncloud->width*ncloud->height << std::endl;
    display_cloud(ncloud, ncenters, normals);
}

void pointcloud_decompressor::display_cloud(pointcloud::Ptr display_cloud,
                                          pcl::PointCloud<pcl::PointXYZ>::Ptr display_centers,
                                          pcl::PointCloud<pcl::Normal>::Ptr display_normals)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer>
            viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);

    // Coloring and visualizing target cloud (red).
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(display_cloud);
    viewer->addPointCloud<point> (display_cloud, rgb, "cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    1, "cloud");

    viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(display_centers, display_normals, 10, 0.05, "normals");

    // Starting visualizer
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

    // Wait until visualizer window is closed.
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
}

void pointcloud_decompressor::read_dict_file(MatrixXf& dict, const std::string& file)
{
    std::ifstream dict_file(file, std::ios::binary);
    int cols = dict.cols();
    int rows = dict.rows();
    dict_file.read((char*)&cols, sizeof(int));
    dict_file.read((char*)&rows, sizeof(int));
    dict.resize(rows, cols);
    float value;
    for (int j = 0; j < dict.cols(); ++j) {
        for (int n = 0; n < dict.rows(); ++n) {
            dict_file.read((char*)&value, sizeof(float));
            dict(n, j) = value;
        }
    }
    dict_file.close();
}

bool pointcloud_decompressor::read_bool(std::ifstream& i, u_char& buffer, int& b)
{
    if (b == 0 || b == 8) {
        i.read((char*)&buffer, sizeof(u_char));
        b = 0;
    }
    bool bit = (buffer >> b) & u_char(1);
    b++;
    return bit;
}

void pointcloud_decompressor::read_from_file(const std::string& file)
{
    std::string rgbfile = file + "rgb.pcdict";
    read_dict_file(RGB_D, rgbfile);
    std::string depthfile = file + "depth.pcdict";
    read_dict_file(D, depthfile);
    std::string code = file + ".pcdcode";

    std::ifstream code_file(code, std::ios::binary);
    int nbr;
    code_file.read((char*)&nbr, sizeof(int)); // number of patches
    code_file.read((char*)&sz, sizeof(int));
    code_file.read((char*)&words_max, sizeof(int));
    code_file.read((char*)&RGB_words_max, sizeof(int));

    S.resize(sz*sz, nbr);
    W.resize(sz*sz, nbr);
    RGB.resize(sz*sz, 3*nbr);
    rotations.resize(nbr);
    means.resize(nbr);
    RGB_means.resize(nbr);

    X.resize(words_max, nbr);
    I.resize(words_max, nbr);
    number_words.resize(nbr);

    RGB_X.resize(RGB_words_max, 3*nbr);
    RGB_I.resize(RGB_words_max, 3*nbr);
    RGB_number_words.resize(3*nbr);

    code_file.read((char*)&dict_size, sizeof(int)); // dictionary size
    code_file.read((char*)&RGB_dict_size, sizeof(int)); // RGB dictionary size
    code_file.read((char*)&res, sizeof(float)); // size of voxels
    float value;
    for (int i = 0; i < S.cols(); ++i) { // means of patches
        for (int n = 0; n < 3; ++n) {
            code_file.read((char*)&value, sizeof(float));
            means[i](n) = value;
        }
    }
    for (int i = 0; i < S.cols(); ++i) { // rotations of patches
        for (int m = 0; m < 3; ++m) {
            for (int n = 0; n < 3; ++n) {
                code_file.read((char*)&value, sizeof(float));
                rotations[i](m, n) = value;
            }
        }
    }
    u_char words;
    for (int i = 0; i < S.cols(); ++i) { // number of words and codes
        code_file.read((char*)&words, sizeof(u_char));
        number_words[i] = words;
        for (int n = 0; n < words; ++n) {
            code_file.read((char*)&value, sizeof(float));
            X(n, i) = value;
        }
    }
    u_char word;
    for (int i = 0; i < S.cols(); ++i) { // dictionary entries used
        for (int n = 0; n < number_words[i]; ++n) {
            code_file.read((char*)&word, sizeof(u_char));
            /*std::cout << I.col(i).transpose() << std::endl;
            std::cout << I.row(n) << std::endl;
            std::cout << n << " " << i << " " << I.rows() << " " << I.cols() << " " << number_words[i] << std::endl;*/
            I(n, i) = int(word);
        }
    }
    for (int i = 0; i < S.cols(); ++i) { // rgb means of patches
        for (int n = 0; n < 3; ++n) {
            code_file.read((char*)&value, sizeof(float));
            RGB_means[i](n) = value;
        }
    }
    for (int i = 0; i < 3*S.cols(); ++i) { // rgb number of words and codes
        code_file.read((char*)&words, sizeof(u_char));
        RGB_number_words[i] = words;
        for (int n = 0; n < words; ++n) {
            code_file.read((char*)&value, sizeof(float));
            RGB_X(n, i) = value;
        }
    }
    for (int i = 0; i < 3*S.cols(); ++i) { // rgb dictionary entries used
        for (int n = 0; n < RGB_number_words[i]; ++n) {
            code_file.read((char*)&word, sizeof(u_char));
            RGB_I(n, i) = word;
        }
        //std::cout << RGB_I(0, i) << " " << RGB_I(0, i+1) << std::endl;
        //std::cout << "-----------" << std::endl;
    }
    u_char buffer = 0;
    int b = 0;
    for (int i = 0; i < S.cols(); ++i) { // masks of patches
        for (int n = 0; n < sz*sz; ++n) {
            W(n, i) = read_bool(code_file, buffer, b);
        }
    }
    code_file.close();
}
