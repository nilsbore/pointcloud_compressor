#ifndef KSVD_DECOMPOSITION_H
#define KSVD_DECOMPOSITION_H

#include <Eigen/Dense>
#include <vector>

class ksvd_decomposition
{
private:
    Eigen::MatrixXf& X;
    Eigen::MatrixXi& I;
    Eigen::MatrixXf& D;
    std::vector<int>& number_words;
    const Eigen::MatrixXf& S;
    const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& W;
    int dict_size;
    int words_max;
    float proj_error;
    float stop_diff;
    int l;
    int n;
    std::vector<std::vector<int>> L;
    std::vector<std::vector<int>> Lk;
    std::vector<int> unused;
public:
    void decompose();
    int compute_code();
    void optimize_dictionary();
    float compute_error();
    void replace_unused();
    void randomize_positions(std::vector<int>& rtn, int m);
    ksvd_decomposition(Eigen::MatrixXf& X, Eigen::MatrixXi& I, Eigen::MatrixXf& D,
                       std::vector<int>& number_words, const Eigen::MatrixXf& S,
                       const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& W,
                       int dict_size, int words_max, float proj_error, float stop_diff);
};

#endif // KSVD_DECOMPOSITION_H
