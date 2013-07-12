#include "pointcloud_compressor.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/octree/octree_impl.h>
#include <pcl/io/pcd_io.h>
#include <vector>
#include <stdint.h>
#include <boost/thread/thread.hpp>

using namespace Eigen;

pointcloud_compressor::pointcloud_compressor(const std::string& filename, float res, int sz) : cloud(new pointcloud), res(res), sz(sz)
{
    if (pcl::io::loadPCDFile<point> (filename, *cloud) == -1)
    {
      PCL_ERROR("Couldn't read file room_scan2.pcd \n");
      return;
    }
    compress_cloud();
    decompress_cloud();
}

void pointcloud_compressor::compute_rotation(Matrix3f& R, const MatrixXf& points)
{
    if (points.cols() < 4) {
        R.setIdentity();
        return;
    }
    JacobiSVD<MatrixXf> svd(points.transpose(), ComputeThinV); // kan ta U ist för transpose?
    Vector3f normal;
    normal << svd.matrixV().block<3, 1>(0, 3);
    normal.normalize();
    if (normal(2) < 0) { // To ensure that normals on a surface are oriented in the same direction
        normal *= -1;
    }
    R.col(0) << normal;
    Vector3f up(0.0f, 0.0f, 1.0f);
    Vector3f forward(1.0f, 0.0f, 0.0f);
    if (fabs(up.dot(normal)) > cos(M_PI/4.0f) || fabs(up.dot(normal)) < cos(3.0f*M_PI/4.0f)) {
        R.col(1) << forward.cross(normal);
    }
    else {
        R.col(1) << up.cross(normal);
    }
    R.col(1).normalize();
    R.col(2) << normal.cross(R.col(1));
}

void pointcloud_compressor::project_points(MatrixXf& rtn, Vector3f& center,
                                           Array<bool, Dynamic, Dynamic>& isSet,
                                           const Matrix3f& R, MatrixXf& points,
                                           Matrix<short, Dynamic, Dynamic>& red,
                                           Matrix<short, Dynamic, Dynamic>& green,
                                           Matrix<short, Dynamic, Dynamic>& blue,
                                           const Matrix<short, Dynamic, Dynamic>& colors)
{
    MatrixXi count(sz, sz);
    count.setZero();
    Vector3f pt;
    Matrix<short, 3, 1> c;
    for (int i = 0; i < points.cols(); ++i) {
        pt = R.transpose()*(points.block<3, 1>(0, i) - center);
        pt(1) += res/2.0f;
        pt(2) += res/2.0f;
        if (pt(1) > res || pt(1) < 0 || pt(2) > res || pt(2) < 0) {
            continue;
        }
        int x = int(float(sz)*pt(1)/res);
        int y = int(float(sz)*pt(2)/res);
        float current_count = count(y, x);
        rtn(y, x) = (current_count*rtn(y, x) + pt(0)) / (current_count + 1);
        c = colors.col(i);
        red(y, x) = short((current_count*float(red(y, x)) + float(c(0))) / (current_count + 1));
        green(y, x) = short((current_count*float(green(y, x)) + float(c(1))) / (current_count + 1));
        blue(y, x) = short((current_count*float(blue(y, x)) + float(c(2))) / (current_count + 1));
        count(y, x) += 1;
    }
    float mn = rtn.mean();
    rtn.array() -= mn;
    center += mn*R.col(0);
    isSet = count.array() > 0;
}

void pointcloud_compressor::compress_cloud()
{
    pcl::octree::OctreePointCloudSearch<point> octree(res);
    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();

    std::vector<point, Eigen::aligned_allocator<point> > centers;
    octree.getOccupiedVoxelCenters(centers);

    float radius = sqrt(3.0f)/2.0f*res;
    //float radius = 1.0f/sqrt(2.0f)*res;

    std::vector<int> index_search;
    std::vector<float> distances;
    for (point center : centers) {
        octree.radiusSearch(center, radius, index_search, distances);
        MatrixXf points(4, index_search.size());
        Matrix<short, Dynamic, Dynamic> colors(3, index_search.size());
        points.row(3).setOnes();
        for (int i = 0; i < index_search.size(); ++i) {
            points(0, i) = cloud->points[index_search[i]].x;
            points(1, i) = cloud->points[index_search[i]].y;
            points(2, i) = cloud->points[index_search[i]].z;
            colors(0, i) = cloud->points[index_search[i]].r;
            colors(1, i) = cloud->points[index_search[i]].g;
            colors(2, i) = cloud->points[index_search[i]].b;
        }
        Eigen::Matrix3f R;
        compute_rotation(R, points);
        MatrixXf rtn(sz, sz);
        Matrix<short, Dynamic, Dynamic> red(sz, sz);
        Matrix<short, Dynamic, Dynamic> green(sz, sz);
        Matrix<short, Dynamic, Dynamic> blue(sz, sz);
        Array<bool, Dynamic, Dynamic> mask(sz, sz);
        Vector3f center_rtn(center.x, center.y, center.z);
        project_points(rtn, center_rtn, mask, R, points, red, green, blue, colors);
        rotations.push_back(R);
        mids.push_back(center_rtn);
        patches.push_back(rtn);
        masks.push_back(mask);
        reds.push_back(red);
        greens.push_back(green);
        blues.push_back(blue);
    }
}

void pointcloud_compressor::decompress_cloud()
{
    int n = patches.size();
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
    for (int i = 0; i < n; ++i) {
        for (int y = 0; y < sz; ++y) { // ROOM FOR SPEEDUP
            for (int x = 0; x < sz; ++x) {
                if (!masks[i](y, x)) {
                    continue;
                }
                pt(0) = patches[i](y, x);
                pt(1) = (float(x) + 0.5f)*res/float(sz);
                pt(2) = (float(y) + 0.5f)*res/float(sz);
                pt = rotations[i]*pt + mids[i];
                int ind = i*sz*sz + y*sz + x;
                ncloud->at(ind).x = pt(0);
                ncloud->at(ind).y = pt(1);
                ncloud->at(ind).z = pt(2);
                ncloud->at(ind).r = reds[i](y, x);
                ncloud->at(ind).g = greens[i](y, x);
                ncloud->at(ind).b = blues[i](y, x);
            }
        }
        ncenters->at(i).x = mids[i](0);
        ncenters->at(i).y = mids[i](1);
        ncenters->at(i).z = mids[i](2);
        normals->at(i).normal_x = rotations[i](0, 0);
        normals->at(i).normal_y = rotations[i](1, 0);
        normals->at(i).normal_z = rotations[i](2, 0);
    }
    display_cloud(ncloud, ncenters, normals);
}

void pointcloud_compressor::display_cloud(pointcloud::Ptr display_cloud,
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

    //viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(display_centers, display_normals, 10, 0.05, "normals");

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
