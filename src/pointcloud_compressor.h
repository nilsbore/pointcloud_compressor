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
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> RGB;
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> W;

    Eigen::MatrixXf D;
    Eigen::MatrixXf X;
    Eigen::MatrixXi I;
    std::vector<int> number_words;

    Eigen::MatrixXf RGB_D;
    Eigen::MatrixXf RGB_X;
    Eigen::MatrixXi RGB_I;
    std::vector<int> RGB_number_words;

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
    void write_to_file(const std::string& file);
    void read_from_file(const std::string& file);
public:
    pointcloud_compressor(const std::string& filename, float res, int sz, int dict_size,
                          int words_max, float proj_error, float stop_diff);
};

#endif // POINTCLOUD_COMPRESSOR_H
