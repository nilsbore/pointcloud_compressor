#include "pointcloud_compressor.h"

#include "ksvd_decomposition.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/octree/octree_impl.h>
#include <pcl/io/pcd_io.h>
#include <stdint.h>
#include <boost/thread/thread.hpp>
//#include <boost/archive/text_oarchive.hpp>
//#include <boost/archive/text_iarchive.hpp>
#include <fstream>

using namespace Eigen;

pointcloud_compressor::pointcloud_compressor(const std::string& filename, float res, int sz, int dict_size,
                                             int words_max, float proj_error, float stop_diff) :
    cloud(new pointcloud), res(res), sz(sz), dict_size(dict_size), RGB_dict_size(200),
    words_max(words_max), RGB_words_max(20), proj_error(proj_error), stop_diff(stop_diff)
{
    if (pcl::io::loadPCDFile<point> (filename, *cloud) == -1)
    {
      PCL_ERROR("Couldn't read file room_scan2.pcd \n");
      return;
    }

    compress();
}

void pointcloud_compressor::compress()
{
    std::cout << "Size of original point cloud: " << cloud->width*cloud->height << std::endl;
    project_cloud();
    std::cout << "Number of patches: " << S.cols() << std::endl;
    compress_depths();
    compress_colors();
    decompress_depths();
    decompress_colors();
    reproject_cloud();
    write_to_file(std::string("test"));
    read_from_file(std::string("test"));
}

void pointcloud_compressor::compute_rotation(Matrix3f& R, const MatrixXf& points)
{
    if (points.cols() < 4) {
        R.setIdentity();
        return;
    }
    JacobiSVD<MatrixXf> svd(points.transpose(), ComputeThinV); // kan ta U ist för transpose?
    Vector3f normal = svd.matrixV().block<3, 1>(0, 3);
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

void pointcloud_compressor::project_points(Vector3f& center, const Matrix3f& R, MatrixXf& points,
                                           const Matrix<short, Dynamic, Dynamic>& colors,
                                           const std::vector<int>& index_search,
                                           int* occupied_indices, int i)
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
        S(ind, i) = (current_count*S(ind, i) + pt(0)) / (current_count + 1);
        c = colors.col(m);
        for (int n = 0; n < 3; ++n) {
            RGB(ind, n*S.cols() + i) = (current_count*RGB(ind, n*S.cols() + i) + float(c(n))) / (current_count + 1);
        }
        count(ind) += 1;
    }
    float mn = S.col(i).mean();
    S.col(i).array() -= mn;
    center += mn*R.col(0); // should this be minus??
    mn = RGB.col(i).mean();
    RGB.col(i).array() -= mn;
    RGB_means[i](0) = mn;
    mn = RGB.col(S.cols() + i).mean();
    RGB.col(S.cols() + i).array() -= mn;
    RGB_means[i](1) = mn;
    mn = RGB.col(2*S.cols() + i).mean();
    RGB.col(2*S.cols() + i).array() -= mn;
    RGB_means[i](2) = mn;
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

    S.resize(sz*sz, centers.size());
    W.resize(sz*sz, centers.size());
    //RGB.resize(3*sz*sz, centers.size());
    RGB.resize(sz*sz, 3*centers.size());
    rotations.resize(centers.size());
    means.resize(centers.size());
    RGB_means.resize(centers.size());

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
        project_points(mid, R, points, colors, index_search, occupied_indices, i);
        rotations[i] = R; // rewrite all this shit to use arrays instead
        means[i] = mid;
    }
    delete[] occupied_indices;
}

void pointcloud_compressor::compress_depths()
{
    ksvd_decomposition ksvd(X, I, D, number_words, S, W, dict_size, words_max, proj_error, stop_diff);
}

void pointcloud_compressor::compress_colors()
{
    Matrix<bool, Dynamic, Dynamic> RGB_W(sz*sz, RGB.cols());
    for (int n = 0; n < 3; ++n) {
        RGB_W.block(0, n*S.cols(), sz*sz, S.cols()) = W;
    }
    ksvd_decomposition(RGB_X, RGB_I, RGB_D, RGB_number_words, RGB,
                       RGB_W, RGB_dict_size, RGB_words_max, 1e3f, 1e6f);
}

void pointcloud_compressor::decompress_depths()
{
    for (int i = 0; i < S.cols(); ++i) {
        S.col(i).setZero();
        for (int k = 0; k < number_words[i]; ++k) {
            S.col(i) += X(k, i)*D.col(I(k, i));
        }
    }
}

void pointcloud_compressor::decompress_colors()
{
    for (int i = 0; i < RGB.cols(); ++i) {
        RGB.col(i).setZero();
        for (int k = 0; k < RGB_number_words[i]; ++k) {
            RGB.col(i) += RGB_X(k, i)*RGB_D.col(RGB_I(k, i));
        }
    }
}

void pointcloud_compressor::reproject_cloud()
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

void pointcloud_compressor::write_dict_file(const MatrixXf& dict, const std::string& file)
{
    std::ofstream dict_file(file, std::ios::binary | std::ios::trunc);
    int cols = dict.cols();
    int rows = dict.rows();
    dict_file.write((char*)&cols, sizeof(int));
    dict_file.write((char*)&rows, sizeof(int));
    float value;
    for (int j = 0; j < dict.cols(); ++j) {
        for (int n = 0; n < dict.rows(); ++n) {
            value = dict(n, j);
            dict_file.write((char*)&value, sizeof(float));
        }
    }
    dict_file.close();
}

void pointcloud_compressor::read_dict_file(MatrixXf& dict, const std::string& file)
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

void pointcloud_compressor::write_bool(std::ofstream& o, u_char& buffer, int& b, bool bit)
{
    if (b == 8) {
        o.write((char*)&buffer, sizeof(u_char));
        buffer = 0;
        b = 0;
    }
    buffer |= u_char(bit) << b;
    b++;
}

bool pointcloud_compressor::read_bool(std::ifstream& i, u_char& buffer, int& b)
{
    if (b == 0 || b == 8) {
        i.read((char*)&buffer, sizeof(u_char));
        b = 0;
    }
    bool bit = (buffer >> b) & u_char(1);
    b++;
    return bit;
}

void pointcloud_compressor::close_write_bools(std::ofstream& o, u_char& buffer)
{
    o.write((char*)&buffer, sizeof(u_char));
}

void pointcloud_compressor::write_to_file(const std::string& file)
{
    std::string rgbfile = file + "rgb.pcdict";
    write_dict_file(RGB_D, rgbfile);
    std::string depthfile = file + "depth.pcdict";
    write_dict_file(D, depthfile);
    std::string code = file + ".pcdcode";

    std::ofstream code_file(code, std::ios::binary | std::ios::trunc);
    int nbr = S.cols();
    code_file.write((char*)&nbr, sizeof(int)); // number of patches
    code_file.write((char*)&sz, sizeof(int));
    code_file.write((char*)&words_max, sizeof(int));
    code_file.write((char*)&RGB_words_max, sizeof(int));
    code_file.write((char*)&dict_size, sizeof(int)); // dictionary size
    code_file.write((char*)&RGB_dict_size, sizeof(int)); // RGB dictionary size
    code_file.write((char*)&res, sizeof(float)); // size of voxels
    float value;
    for (int i = 0; i < S.cols(); ++i) { // means of patches
        for (int n = 0; n < 3; ++n) {
            value = means[i](n);
            code_file.write((char*)&value, sizeof(float));
        }
    }
    for (int i = 0; i < S.cols(); ++i) { // rotations of patches
        for (int m = 0; m < 3; ++m) {
            for (int n = 0; n < 3; ++n) {
                value = rotations[i](m, n);
                code_file.write((char*)&value, sizeof(float));
            }
        }
    }
    u_char words;
    for (int i = 0; i < S.cols(); ++i) { // number of words and codes
        words = number_words[i];
        code_file.write((char*)&words, sizeof(u_char));
        for (int n = 0; n < words; ++n) {
            value = X(n, i);
            code_file.write((char*)&value, sizeof(float));
        }
    }
    u_char word;
    for (int i = 0; i < S.cols(); ++i) { // dictionary entries used
        for (int n = 0; n < number_words[i]; ++n) {
            word = I(n, i);
            code_file.write((char*)&word, sizeof(u_char));
        }
    }
    for (int i = 0; i < S.cols(); ++i) { // rgb means of patches
        for (int n = 0; n < 3; ++n) {
            value = RGB_means[i](n);
            code_file.write((char*)&value, sizeof(float));
        }
    }
    for (int i = 0; i < 3*S.cols(); ++i) { // rgb number of words and codes
        words = RGB_number_words[i];
        code_file.write((char*)&words, sizeof(u_char));
        for (int n = 0; n < words; ++n) {
            value = RGB_X(n, i);
            code_file.write((char*)&value, sizeof(float));
        }
    }
    for (int i = 0; i < 3*S.cols(); ++i) { // rgb dictionary entries used
        for (int n = 0; n < RGB_number_words[i]; ++n) {
            word = RGB_I(n, i);
            code_file.write((char*)&word, sizeof(u_char));
        }
    }
    u_char buffer = 0;
    int b = 0;
    for (int i = 0; i < S.cols(); ++i) { // masks of patches
        for (int n = 0; n < sz*sz; ++n) {
            write_bool(code_file, buffer, b, W(n, i));
        }
    }
    close_write_bools(code_file, buffer);
    code_file.close();
}

void pointcloud_compressor::read_from_file(const std::string& file)
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
        Array<bool, Dynamic, Dynamic> old = W.col(i);
        for (int n = 0; n < sz*sz; ++n) {
            W(n, i) = read_bool(code_file, buffer, b);
        }
        std::cout << "---------------------" << std::endl;
        std::cout << W.col(i).transpose() << std::endl;
        std::cout << old.transpose() << std::endl;
    }
    code_file.close();
}
