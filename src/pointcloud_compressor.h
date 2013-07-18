#ifndef POINTCLOUD_COMPRESSOR_H
#define POINTCLOUD_COMPRESSOR_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>

class pointcloud_compressor
{
public:
    typedef pcl::PointXYZRGB point;
    typedef pcl::PointCloud<point> pointcloud;
private:
    pointcloud::Ptr cloud;
    float res;
    int sz;
    int dict_sz;
    int words_max;
    float proj_err;
    float end_diff;
    std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f> > rotations;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > mids;
    std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf> > patches;
    std::vector<Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic> > > reds;
    std::vector<Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic> > > greens;
    std::vector<Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic> > > blues;
    std::vector<short> meanReds;
    std::vector<short> meanGreens;
    std::vector<short> meanBlues;
    std::vector<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> > > masks;
    Eigen::MatrixXf D;
    Eigen::MatrixXf X;
    Eigen::MatrixXi I;
    std::vector<int> nbr_bases;
public:
    void project_cloud();
    void display_cloud(pointcloud::Ptr,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr,
                       pcl::PointCloud<pcl::Normal>::Ptr);
    void compute_rotation(Eigen::Matrix3f&, const Eigen::MatrixXf&);
    void project_points(Eigen::MatrixXf&, Eigen::Vector3f&,
                        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>&,
                        const Eigen::Matrix3f&, Eigen::MatrixXf&,
                        Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic>&,
                        Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic>&,
                        Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic>&,
                        const Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic>&,
                        const std::vector<int>&, int*, int);
    void compress_cloud();
    void reconstruct_patches();
    void reconstruct_cloud();
    void get_random_patches(std::vector<int>&, int);
    pointcloud_compressor(const std::string&, float, int, int, int, float, float);
};

#endif // POINTCLOUD_COMPRESSOR_H
