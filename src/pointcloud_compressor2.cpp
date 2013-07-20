#include "pointcloud_compressor2.h"

#include "ksvd_decomposition.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/octree/octree_impl.h>
#include <pcl/io/pcd_io.h>
#include <stdint.h>
#include <boost/thread/thread.hpp>

using namespace Eigen;

pointcloud_compressor::pointcloud_compressor(const std::string& filename, float res, int sz, int dict_sz,
                                             int words_max, float proj_err, float end_diff) :
    cloud(new pointcloud), res(res), sz(sz), dict_sz(dict_sz), words_max(words_max), proj_err(proj_err), end_diff(end_diff)
{
    if (pcl::io::loadPCDFile<point> (filename, *cloud) == -1)
    {
      PCL_ERROR("Couldn't read file room_scan2.pcd \n");
      return;
    }

    compress();
}

void compress()
{
    std::cout << "Size of original point cloud: " << cloud->width*cloud->height << std::endl;
    project_cloud();
    std::cout << "Number of patches: " << S.cols() << std::endl;
    compress_depths();
    compress_colors();
    decompress_depths();
    decompress_colors();
    reproject_cloud();
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
    Vector3f x(1.0f, 0.0f, 0.0f);
    Vector3f y(0.0f, 1.0f, 0.0f);
    Vector3f z(0.0f, 0.0f, 1.0f);
    if (fabs(normal(0)) > fabs(normal(1)) && fabs(normal(0)) > fabs(normal(2))) { // pointing in x dir
        if (normal(0) < 0) {
            normal *= -1;
        }
        R.col(0) = normal;
        R.col(1) = z.cross(normal);
    }
    else if (fabs(normal(1)) > fabs(normal(0)) && fabs(normal(1)) > fabs(normal(2))) { // pointing in y dir
        if (normal(1) < 0) {
            normal *= -1;
        }
        R.col(0) = normal;
        R.col(1) = x.cross(normal);
    }
    else { // pointing in z dir
        if (normal(2) < 0) {
            normal *= -1;
        }
        R.col(0) = normal;
        R.col(1) = y.cross(normal);
    }
    R.col(1).normalize();
    R.col(2) = normal.cross(R.col(1));
}

void pointcloud_compressor::project_points(Vector3f& center,
                                           const Matrix3f& R, MatrixXf& points,
                                           const Matrix<short, Dynamic, Dynamic>& colors,
                                           const std::vector<int>& index_search,
                                           int* occupied_indices,
                                           int i)
{
    ArrayXi count(sz*sz);
    count.setZero();
    Vector3f pt;
    Matrix<short, 3, 1> c;
    int x, y, ind;
    for (int m = 0; m < points.cols(); ++m) {
        if (occupied_indices[index_search[m]]) {
            continue;
        }
        pt = R.transpose()*(points.block<3, 1>(0, m) - center);
        pt(1) += res/2.0f;
        pt(2) += res/2.0f;
        if (pt(1) > res || pt(1) < 0 || pt(2) > res || pt(2) < 0) {
            continue;
        }
        occupied_indices[index_search[m]] = 1;
        x = int(float(sz)*pt(1)/res);
        y = int(float(sz)*pt(2)/res);
        ind = sz*x + y;
        float current_count = count(ind);
        //rtn(y, x) = (current_count*rtn(y, x) + pt(0)) / (current_count + 1);
        S(ind, i) = (current_count*S(ind, i) + pt(0)) / (current_count + 1);
        c = colors.col(m);
        for (int m = 0; m < 3; ++m) {
            RGB(m*sz*sz + ind, i) = short((current_count*float(RGB(m*sz*sz + ind, i)) + float(c(0))) / (current_count + 1));
        }
        count(ind) += 1;
    }
    float mn = S.col(i).mean();
    S.col(i).array() -= mn;
    center += mn*R.col(0); // should this be minus??
    //isSet = count.array() > 0;
    W.col(i) = count > 0;
    //S.col(j).array() *= isSet.cast<float>(); // this is mostly for debugging
}

void pointcloud_compressor::project_cloud()
{
    pcl::octree::OctreePointCloudSearch<point> octree(res);
    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();

    std::vector<point, Eigen::aligned_allocator<point> > centers;
    octree.getOccupiedVoxelCenters(centers);

    float radius = sqrt(3.0f)/2.0f*res; // radius of the sphere encompassing the voxels

    std::vector<int> index_search;
    std::vector<float> distances;
    Eigen::Matrix3f R;
    Vector3f mid;
    int* occupied_indices = new int[cloud->width*cloud->height]();
    point center;
    for (int i = 0; i < centers.size(); ++i) {
        center = centers[i];
        octree.radiusSearch(center, radius, index_search, distances);
        MatrixXf points(4, index_search.size());
        Matrix<short, Dynamic, Dynamic> colors(3, index_search.size());
        points.row(3).setOnes();
        for (int m = 0; m < index_search.size(); ++m) {
            points(0, m) = cloud->points[index_search[m]].x;
            points(1, m) = cloud->points[index_search[m]].y;
            points(2, m) = cloud->points[index_search[m]].z;
            colors(0, m) = cloud->points[index_search[m]].r;
            colors(1, m) = cloud->points[index_search[m]].g;
            colors(2, m) = cloud->points[index_search[m]].b;
        }
        compute_rotation(R, points);
        mid = Vector3f(center.x, center.y, center.z);
        project_points(mid, R, points, colors, index_search, occupied_indices, j);
        rotations.push_back(R); // rewrite all this shit to use arrays instead
        mids.push_back(mid);
    }
    delete[] occupied_indices;
}

void pointcloud_compressor::compress_depths()
{
    ksvd_decomposition(X, I, D, number_words, S, W, dict_size, words_max, proj_error, stop_diff);
}

void pointcloud_compressor::compress_colors()
{
    //ksvd_decomposition(X, I, D, number_words, S, W, dict_size, words_max, proj_error, stop_diff);
}

void pointcloud_compressor::decompress_depths()
{
    /*VectorXf s(sz*sz);
    for (int i = 0; i < patches.size(); ++i) {
        s.setZero();
        for (int k = 0; k < nbr_bases[i]; ++k) {
            s += X(k, i)*D.col(I(k, i));
        }
        patches[i] = MatrixXf::Map(s.data(), sz, sz);
    }*/
}

void pointcloud_compressor::decompress_colors()
{
    /*VectorXf s(sz*sz);
    for (int i = 0; i < patches.size(); ++i) {
        s.setZero();
        for (int k = 0; k < nbr_bases[i]; ++k) {
            s += X(k, i)*D.col(I(k, i));
        }
        patches[i] = MatrixXf::Map(s.data(), sz, sz);
    }*/
}

void pointcloud_compressor::reproject_cloud()
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
    int counter = 0;
    for (int i = 0; i < n; ++i) {
        for (int y = 0; y < sz; ++y) { // ROOM FOR SPEEDUP
            for (int x = 0; x < sz; ++x) {
                if (!masks[i](y, x)) {
                    continue;
                }
                pt(0) = patches[i](y, x);
                pt(1) = (float(x) + 0.5f)*res/float(sz) - res/2.0f;
                pt(2) = (float(y) + 0.5f)*res/float(sz) - res/2.0f;
                pt = rotations[i]*pt + mids[i];
                ncloud->at(counter).x = pt(0);
                ncloud->at(counter).y = pt(1);
                ncloud->at(counter).z = pt(2);
                ncloud->at(counter).r = reds[i](y, x) + meanReds[i];
                ncloud->at(counter).g = greens[i](y, x) + meanGreens[i];
                ncloud->at(counter).b = blues[i](y, x) + meanBlues[i];
                ++counter;
            }
        }
        ncenters->at(i).x = mids[i](0);
        ncenters->at(i).y = mids[i](1);
        ncenters->at(i).z = mids[i](2);
        normals->at(i).normal_x = rotations[i](0, 0);
        normals->at(i).normal_y = rotations[i](1, 0);
        normals->at(i).normal_z = rotations[i](2, 0);
    }
    ncloud->resize(counter);
    std::cout << "Size of transformed point cloud: " << ncloud->width*ncloud->height << std::endl;
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
