#ifndef DICTIONARY_REPRESENTATION_H
#define DICTIONARY_REPRESENTATION_H

#include <pcl/point_types.h>
#include <vector>
#include <fstream>

class dictionary_representation
{
protected:
    float res;
    int sz;
    int dict_size;
    int words_max;
    int RGB_dict_size;
    int RGB_words_max;

    std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf> > rotations;
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

    bool read_bool(std::ifstream& i, u_char& buffer, int& b);
    void read_dict_file(Eigen::MatrixXf& dict, const std::string& file);
    void read_from_file(const std::string& file);

    void write_bool(std::ofstream& o, u_char& buffer, int& b, bool bit);
    void close_write_bools(std::ofstream& o, u_char& buffer);
    void write_dict_file(const Eigen::MatrixXf& dict, const std::string& file);
    void write_to_file(const std::string& file);
public:
    dictionary_representation();
    dictionary_representation(float res, int sz, int dict_size, int words_max,
                              int RGB_dict_size, int RGB_words_max);
};

#endif // DICTIONARY_REPRESENTATION_H
