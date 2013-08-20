#ifndef ORTHOGONAL_MATCHING_PURSUIT_H
#define ORTHOGONAL_MATCHING_PURSUIT_H

#include <Eigen/Dense>

class orthogonal_matching_pursuit
{
private:
    const Eigen::MatrixXf& D;
    int words_max;
    float proj_error;
public:
    void match_vector(Eigen::VectorXf Xi, Eigen::VectorXf Ii,
                      const Eigen::VectorXf& s, const Eigen::VectorXf& mask);
    orthogonal_matching_pursuit(const Eigen::MatrixXf &D, int words_max, float proj_error);
};

#endif // ORTHOGONAL_MATCHING_PURSUIT_H
