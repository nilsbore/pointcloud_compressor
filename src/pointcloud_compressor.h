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
    int dict_size;
    int words_max;
    float proj_error;
    float stop_diff;

    std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f> > rotations;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > means;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > RGB_means;

    Eigen::MatrixXf S;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> R;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> G;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> B;
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> W;

    Eigen::MatrixXf D;
    Eigen::MatrixXf X;
    Eigen::MatrixXi I;
    std::vector<int> number_words;

    Eigen::MatrixXf R_D;
    Eigen::MatrixXf R_X;
    Eigen::MatrixXi R_I;
    std::vector<int> R_number_words;

    Eigen::MatrixXf G_D;
    Eigen::MatrixXf G_X;
    Eigen::MatrixXi G_I;
    std::vector<int> G_number_words;

    Eigen::MatrixXf B_D;
    Eigen::MatrixXf B_X;
    Eigen::MatrixXi B_I;
    std::vector<int> B_number_words;

    void compress();
    void compute_rotation(Eigen::Matrix3f& R, const Eigen::MatrixXf& points);
    void project_points(Eigen::Vector3f& center, const Eigen::Matrix3f& R, Eigen::MatrixXf& points,
                        const Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic>& colors,
                        const std::vector<int>& index_search, int* occupied_indices, int i);
    void project_cloud();
    void compress_depths();
    void compress_colors();
    void decompress_depths();
    void decompress_colors();
    void reproject_cloud();
    void display_cloud(pointcloud::Ptr display_cloud,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr display_centers,
                       pcl::PointCloud<pcl::Normal>::Ptr display_normals);
public:
    pointcloud_compressor(const std::string& filename, float res, int sz, int dict_size,
                          int words_max, float proj_error, float stop_diff);
};

#endif // POINTCLOUD_COMPRESSOR_H
