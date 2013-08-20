#include "orthogonal_matching_pursuit.h"

using namespace Eigen;

// sending the dictionary in here might be bad for parallelization
orthogonal_matching_pursuit::orthogonal_matching_pursuit(const MatrixXf& D,
                                                         int words_max,
                                                         float proj_error) :
    D(D), words_max(words_max), proj_error(proj_error)
{
    //A = D.transpose()*D; // it's assumed that || d_k || = 1
}

void orthogonal_matching_pursuit::match_vector(VectorXf Xi, VectorXf Ii, const VectorXf& s, const VectorXf& mask)
{
    int k = 0;
    Xi.resize(words_max);
    Ii.resize(words_max);
    VectorXf bk;
    VectorXf vk;
    VectorXf f = s.array()*mask.array();
    VectorXf res = f;
    double alphak, beta;
    MatrixXf Akinv;
    ArrayXf weights(D.cols());
    VectorXf norms(words_max);
    VectorXf xk(D.rows());

    int ind;
    // Compute all of the projections
    for (k = 0; k < words_max; ++k) {

        // could be computed recursively
        weights = res.transpose()*D; // if it weren't for the weights, we would be able to do this recursively
        for (int m = 0; m < k; ++m) {
            weights(Ii(m)) = 0;
        }
        weights.abs().maxCoeff(&ind);
        Xi(k) = weights(ind);
        Ii(k) = ind;

        if (k > 0) {
            // hitta korrelationen med de tidigare

            Akinv.conservativeResize(k, k);
            beta = 1/(1 - vk.transpose()*bk);
            Akinv.block(0, 0, k-1, k-1) += beta*bk*bk.transpose();
            Akinv.col(k-1).head(k-1) = -beta*bk;
            Akinv.row(k-1).head(k-1) = -beta*bk.transpose();
            Akinv(k-1, k-1) = beta;

            vk.conservativeResize(k);
            xk = mask.array()*D.col(ind).array();
            norms(k) = xk.norm();
            for (int m = 0; m < k; ++m) {
                vk(m) = 1.0f / (norms(k) * norms(Ii(ind))) * xk.transpose()*D.col(Ii(m));
            }

            bk = Akinv*vk;

            alphak = beta*Xi(k); // assuming || x_k+1 || = 1
            Xi(k) = alphak;
            Xi.head(k) -= alphak*bk; // head på bk?
        }

        res = f;
        for (int m = 0; m <= k; ++m) {
            res.array() -= Xi(m)*mask.array()*D.col(Ii(m)).array();
        }
    }
}

/*
void foo()
{

    mask = W.col(i).cast<float>();
    residual = mask*S.col(i).array();
    int k;
    for (k = 0; k < words_max; ++k) {
        if (residual.squaredNorm() < proj_error) {
            break;
        }
        weights = residual.transpose()*D;
        for (int m = 0; m < k; ++m) {
            weights(I(m, i)) = 0;
        }
        weights.abs().maxCoeff(&ind);
        X(k, i) = weights(ind);
        I(k, i) = ind;
        residual.array() -= X(k, i)*mask*D.col(ind).array();
        L[ind].push_back(i); // this is done because we do it in the opposite way her
        Lk[ind].push_back(k); // when recreating, focus lies on recreating the patches fast
    }
    number_words[i] = k;
    mean_words += k;
}
*/
