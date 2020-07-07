#ifndef GREENKHORN
#define GREENKHORN

#include <Eigen/Dense>
#include <vector>
#include "utils/clustering.hpp"
#include <math.h> // assert

using Eigen::MatrixXd;
using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Eigen::Ref;
using Eigen::VectorXd;

#ifdef NDEBUGCERR
#include <iostream> // debug
#endif

namespace greenkhorn
{
    void roundingTrick(MatrixXdR &matrix, const VectorXd &r, const VectorXd &c)
    {
        // Despite giving the same code in theory, it's much faster to use colwise than multiply with the vector "asDiagonal".
        matrix.array().colwise() *= ((r.array() / matrix.rowwise().sum().array()).min(1.));
        matrix.array().rowwise() *= (c.array() / matrix.colwise().sum().array().transpose()).min(1.).transpose();
        VectorXd row_residual{r - matrix.rowwise().sum()};
        matrix += (row_residual * (c - matrix.colwise().sum().transpose()).transpose() / row_residual.lpNorm<1>());
    }

    enum struct KHORN_TYPE
    {
        SIN,
        GREEN,
    };

    enum struct KHORN_ERROR
    {
        SUCCESS,
        ITER_MAX,
    };

    std::string khornErrorMessage(KHORN_ERROR k)
    {
        switch (k)
        {
        case KHORN_ERROR::SUCCESS:
            return std::string("0: Khorn converged successfully.");
        case KHORN_ERROR::ITER_MAX:
            return std::string("1: Maximum number of iterations reached. Precision is probably below machine's precision.");
        default:
            return std::string("Unknown error code.");
        }
    }

    struct khorn_return
    { // Lot of copying, but will be useful for debugging purposes
        VectorXd u;
        VectorXd v;
        std::vector<double> errors;
        MatrixXdR transport_plan;
        const MatrixXdR cost_matrix;
        KHORN_ERROR k_error;
        KHORN_TYPE k_type;
        double duality_gap{0.};

        // Enable to set the duality gap before it is rounded
        double dualityGap(const VectorXd &r, const VectorXd &c, double eta)
        {
            return (
                // primal min:
                (cost_matrix.array() * transport_plan.array()).sum() + eta * (transport_plan.array() * transport_plan.array().log()).sum()
                // dual max
                - (u.dot(r) + v.dot(c) - transport_plan.sum()));
        }

        double transportCost()
        {
            return (transport_plan.array() * cost_matrix.array()).sum();
        };

        double costMatrixMax()
        {
            return cost_matrix.maxCoeff();
        };

        std::string errorMessage()
        {
            return khornErrorMessage(k_error);
        }
    };

    inline void rowColumnScaling(const VectorXd &u, const VectorXd &v, const MatrixXdR &Gibbs, MatrixXdR &out)
    {
        // Eigen will optimize away this
        out = u.array().exp().matrix().asDiagonal() * Gibbs * v.array().exp().matrix().asDiagonal();
    }

    inline void rowScaling(const VectorXd &u, const MatrixXdR &Gibbs, MatrixXdR &out)
    {
        out = u.array().exp().matrix().asDiagonal() * Gibbs;
    }

    inline void colScaling(const VectorXd &v, const MatrixXdR &Gibbs, MatrixXdR &out)
    {
        out = Gibbs * v.array().exp().matrix().asDiagonal();
    }

    inline void singleRowScaling(const Eigen::Index index, const double u_value, const MatrixXdR &Gibbs, MatrixXdR &out)
    {
        out.row(index) = Gibbs.row(index) * exp(u_value);
    }

    inline void singleColScaling(const Eigen::Index index, const double v_value, const MatrixXdR &Gibbs, MatrixXdR &out)
    {
        out.col(index) = Gibbs.col(index) * exp(v_value);
    }

    khorn_return greenkhorn(const MatrixXdR &C, const VectorXd &r, const VectorXd &c, double eps, double eta,
                            VectorXd &u, VectorXd &v)
    {
        // Could store two Gibbs matrix, one in row order and the other in columns order?
        const MatrixXdR Gibbs{(-C.array() / eta).exp()};

        // const values to speed up computation, at the price of some memory
        const VectorXd r_log{r.array().log().matrix()};
        const VectorXd c_log{c.array().log().matrix()};
        const VectorXd r_log_prim{-r + (r.array() * r_log.array()).matrix()};
        const VectorXd c_log_prim{-c + (c.array() * c_log.array()).matrix()};

        const auto N{
            std::max(C.rows(), C.cols())};
        const double R{
            C.maxCoeff() / eta + log(N) - 2 * log(std::min(r.minCoeff(), c.minCoeff()))};
        const int ITER_MAX{static_cast<int>(2. + 112. * N * R / eps)};

        // stored values of the comparaison
        MatrixXdR B_t{Gibbs};
        VectorXd r_B{B_t.rowwise().sum()};
        VectorXd c_B{B_t.colwise().sum().transpose()};
        VectorXd::Index r_ind{};
        VectorXd::Index c_ind{};
        double r_value{};
        double c_value{};

        // Values updated in the loop
        double E_t{(r_B - r).lpNorm<1>() + (c_B - c).lpNorm<1>()};
        int n_iter{0};

        std::vector<double> errors;
        errors.push_back(E_t);

        while ((n_iter < ITER_MAX) && (E_t > eps))
        {
            r_value = (r_B.array() - r.array() * r_B.array().log() + r_log_prim.array()).matrix().maxCoeff(&r_ind);
            c_value = (c_B.array() - c.array() * c_B.array().log() + c_log_prim.array()).matrix().maxCoeff(&c_ind);
#ifdef NDEBUGCERR
            std::cerr << "r_value: " << r_value << std::endl;
            std::cerr << "c_value: " << c_value << std::endl;
#endif
            if (r_value > c_value)
            { // Update rows
                c_B -= B_t.row(r_ind).transpose();
                // E_t -= abs(r_B(r_ind) - r(r_ind));

                u(r_ind) += r_log(r_ind) - log(r_B(r_ind));
                singleRowScaling(r_ind, u(r_ind), Gibbs, B_t);

                r_B(r_ind) = B_t.row(r_ind).sum();
                c_B += B_t.row(r_ind).transpose();
                // E_t += abs(r_B(r_ind) - r(r_ind));
            }
            else
            { // Update cols
                r_B -= B_t.col(c_ind);

                v(c_ind) += c_log(c_ind) - log(c_B(c_ind));
                singleColScaling(c_ind, v(c_ind), Gibbs, B_t);

                c_B(c_ind) = B_t.col(c_ind).sum();
                r_B += B_t.col(c_ind);
            }

            E_t = (r_B - r).lpNorm<1>() + (c_B - c).lpNorm<1>();

#ifdef NDEBUGCERR
            std::cerr << "r_B: " << r_B.transpose() << std::endl;
            std::cerr << "Computed sum of rows: " << B_t.rowwise().sum().transpose() << std::endl;
            std::cerr << "c_B: " << c_B.transpose() << std::endl;
            std::cerr << "Computed sum of cols: " << B_t.colwise().sum() << std::endl;
#endif

            assert((r_B - B_t.rowwise().sum()).cwiseAbs().sum() < 1e-8);             // debug
            assert((c_B - B_t.colwise().sum().transpose()).cwiseAbs().sum() < 1e-8); // debug

            errors.push_back(E_t);
            ++n_iter;
        }

        KHORN_ERROR error_code;
        if (n_iter == ITER_MAX)
            error_code = KHORN_ERROR::ITER_MAX;
        else
            error_code = KHORN_ERROR::SUCCESS;

        const MatrixXdR &transport_plan{B_t}; // Alias to B_t
        errors.shrink_to_fit();               // return unused allocated memory to the OS
        khorn_return result{u, v, errors, transport_plan, C, error_code, KHORN_TYPE::GREEN};
        result.duality_gap = result.dualityGap(r, c, eta);
        return result;
    }

    khorn_return sinkhorn(const MatrixXdR &C, const VectorXd &r, const VectorXd &c, double eps, double eta,
                          VectorXd &u, VectorXd &v)
    {
        // Could store two Gibbs matrix, one in row order and the other in columns order?
        const MatrixXdR Gibbs{(-C.array() / eta).exp()};

        const double R{
            -log(Gibbs.minCoeff() * std::min(r.minCoeff(), c.minCoeff()))};
        const int ITER_MAX{static_cast<int>(2. + 4. * R / eps)};

        // const values to speed up computation, at the price of some memory
        const VectorXd r_log{r.array().log().matrix()};
        const VectorXd c_log{c.array().log().matrix()};

        // Values updated in the loop
        MatrixXdR B_t(Gibbs);
        VectorXd r_B{B_t.rowwise().sum()};
        VectorXd c_B{B_t.colwise().sum().transpose()};

        double E_t{(r_B - r).lpNorm<1>() + (c_B - c).lpNorm<1>()};
        int n_iter{0};
        std::vector<double> errors;
        errors.push_back(E_t);

        while ((n_iter < ITER_MAX) && E_t > eps)
        {
            if (n_iter % 2 == 0)
            {
                u.array() += r_log.array() - r_B.array().log();
            }
            else
            {
                v.array() += c_log.array() - c_B.array().log();
            }

            rowColumnScaling(u, v, Gibbs, B_t);
            r_B = B_t.rowwise().sum();
            c_B = B_t.colwise().sum().transpose();

            E_t = (r_B - r).lpNorm<1>() + (c_B - c).lpNorm<1>();

            errors.push_back(E_t);
            ++n_iter;
        }

        KHORN_ERROR error_code;
        if (n_iter == ITER_MAX)
            error_code = KHORN_ERROR::ITER_MAX;
        else
            error_code = KHORN_ERROR::SUCCESS;

        const MatrixXdR &transport_plan{B_t}; // Alias to B_t
        errors.shrink_to_fit();               // return unused allocated memory to the OS

        khorn_return result{u, v, errors, transport_plan, C, error_code, KHORN_TYPE::SIN};
        result.duality_gap = result.dualityGap(r, c, eta);
        return result;
    }

    template <KHORN_TYPE k>
    khorn_return khorn(const MatrixXdR &C, const VectorXd &r, const VectorXd &c, double eps, double eta,
                       VectorXd &initial_u, VectorXd &initial_v)
    {
#ifdef NDEBUGCERR
        std::cerr << "Calling Khorn with rows: " << C.rows() << " and cols: " << C.cols() << std::endl;
        std::cerr << "Histograms are: " << r.transpose() << "\n"
                  << c.transpose() << std::endl;
        std::cerr << "Regularization is: " << eta << " and precision is: " << eps << std::endl;
#endif

        // TODO: Factorize these parts
        switch (k)
        {
        case KHORN_TYPE::GREEN:
            return greenkhorn(C, r, c, eps, eta, initial_u, initial_v);
            break;

        case KHORN_TYPE::SIN:
            return sinkhorn(C, r, c, eps, eta, initial_u, initial_v);
            break;
        }
    }

    template <KHORN_TYPE k>
    khorn_return khorn(const MatrixXdR &C, const VectorXd &r, const VectorXd &c, double eps, double eta)
    {
        VectorXd initial_u{VectorXd::Zero(C.rows())};
        VectorXd initial_v{VectorXd::Zero(C.cols())};
        return khorn<k>(C, r, c, eps, eta, initial_u, initial_v);
    }

    template <KHORN_TYPE k>
    khorn_return khorn(const MatrixXdR &C, const VectorXd &r, const VectorXd &c, double eps,
                       VectorXd &initial_u, VectorXd &initial_v)
    {
        // TODO: check marginals are in the probability simplex
        const auto n{
            std::min(C.rows(), C.cols())};
        const double eta{
            eps / (4 * log(n + 1))};
        const double eps_prime{
            eps / (8 * C.maxCoeff())};

#ifdef NDEBUGCERR
        std::cerr << "Calling Khorn with cost Matrix: \n"
                  << C << std::endl;
        std::cerr << "Before conversion, histograms are: \n"
                  << r.transpose() << "\n"
                  << c.transpose() << std::endl;
#endif

        khorn_return result = khorn<k>(C,
                                       (1 - eps_prime / 8) * r.array() + (eps_prime / (8 * C.rows())),
                                       (1 - eps_prime / 8) * c.array() + (eps_prime / (8 * C.cols())),
                                       eps_prime / 2, eta,
                                       initial_u, initial_v);
        roundingTrick(result.transport_plan, r, c);
        return result;
    }

    template <KHORN_TYPE k>
    khorn_return khorn(const MatrixXdR &C, const VectorXd &r, const VectorXd &c, double eps)
    {
        VectorXd initial_u{VectorXd::Zero(C.rows())};
        VectorXd initial_v{VectorXd::Zero(C.cols())};
        return khorn<k>(C, r, c, eps, initial_u, initial_v);
    }

    MatrixXdR pairwiseSquaredDistance(const MatrixXdR &X, const MatrixXdR &Y)
    {
        MatrixXdR C(X.rows(), Y.rows());
        for (Eigen::Index i = 0; i < X.rows(); i++)
        {
            C.row(i) = (Y.rowwise() - X.row(i)).rowwise().squaredNorm().transpose();
        }

        return C;
    }

    template <KHORN_TYPE k>
    khorn_return entropicWasserstein(const MatrixXdR &X, const MatrixXdR &Y,
                                     const VectorXd &r, const VectorXd &c,
                                     double precision,
                                     VectorXd &initial_u, VectorXd &initial_v)
    {
        MatrixXdR C{pairwiseSquaredDistance(X, Y)};
        return khorn<k>(C, r.array() / r.sum(), c.array() / c.sum(), precision);
    }

    template <KHORN_TYPE k>
    khorn_return entropicWasserstein(const MatrixXdR &X, const MatrixXdR &Y,
                                     const VectorXd &r, const VectorXd &c,
                                     double precision)
    {
        VectorXd initial_u{VectorXd::Zero(X.rows())};
        VectorXd initial_v{VectorXd::Zero(Y.rows())};
        return entropicWasserstein<k>(X, Y, r, c, precision, initial_u, initial_v);
    }

    template <KHORN_TYPE k>
    khorn_return entropicWassersteinWarmStart(const Ref<const MatrixXdR> &X, const Ref<const MatrixXdR> &Y, double precision,
                                              VectorXd &initial_u, VectorXd &initial_v)
    {
        return entropicWasserstein<k>(X, Y, VectorXd::Ones(X.rows()), VectorXd::Ones(Y.rows()), precision, initial_u, initial_v);
    }

    template <KHORN_TYPE k>
    khorn_return entropicWasserstein(const Ref<const MatrixXdR> &X, const Ref<const MatrixXdR> &Y, double precision)
    {
        VectorXd initial_u{VectorXd::Zero(X.rows())};
        VectorXd initial_v{VectorXd::Zero(Y.rows())};
        return entropicWassersteinWarmStart<k>(X, Y, precision, initial_u, initial_v);
    }

    template <KHORN_TYPE k>
    khorn_return entropicWassersteinAddPoint(const Ref<const MatrixXdR> &C, const Ref<const MatrixXdR> &X, const Ref<const MatrixXdR> &Y,
                                             const Ref<const VectorXd> &x, const Ref<const VectorXd> &y, double precision,
                                             const Ref<const VectorXd> &initial_u, const Ref<const VectorXd> &initial_v)
    {
        // This is probably not the most efficient way to recopy the whole matrix; just for test purpose for now
        // Probably easier to create a class with its own X, Y, C, r, c, and so on.
        // TODO: cleanify this if used more
        MatrixXd new_C{C.rows() + 1, C.cols() + 1};
        new_C.topLeftCorner(C.rows(), C.cols()) = C;
        new_C.bottomLeftCorner(1, C.rows()) = pairwiseSquaredDistance(X, y.transpose()).transpose();
        new_C.topRightCorner(C.cols(), 1) = pairwiseSquaredDistance(Y, x.transpose());
        new_C(C.rows(), C.cols()) = (x - y).squaredNorm();

        VectorXd u{initial_u.rows() + 1}, v{initial_v.rows() + 1};
        u.head(initial_u.rows()) = initial_u;
        v.head(initial_v.rows()) = initial_v;

        auto r {VectorXd::Ones(X.rows() + 1).array()};
        auto c {VectorXd::Ones(Y.rows() + 1).array()};

        return khorn<k>(new_C, r.array() / r.sum(), c.array() / c.sum(), precision);
    }

    template <KHORN_TYPE k>
    khorn_return entropicWassersteinKMeans(const Ref<const MatrixXdR> &X, const Ref<const MatrixXdR> &Y, double precision)
    {
        // Run KMeans++ on each point cloud
        auto [X_centroids_ids, X_weights]{clustering::assignmentToCount(clustering::kmeanspp(X, X.rows(), precision / 3))};
        auto [Y_centroids_ids, Y_weights]{clustering::assignmentToCount(clustering::kmeanspp(Y, Y.rows(), precision / 3))};

        // Need to wait Eigen 3.3.9 to be able to take slices
        MatrixXdR X_centroids(X_centroids_ids.rows(), X.cols());
        for (Eigen::Index i = 0; i < X_centroids_ids.rows(); i++)
        {
            X_centroids.row(i) = X.row(X_centroids_ids(i));
        }

        MatrixXdR Y_centroids(Y_centroids_ids.rows(), Y.cols());
        for (Eigen::Index i = 0; i < Y_centroids_ids.rows(); i++)
        {
            Y_centroids.row(i) = Y.row(Y_centroids_ids(i));
        }

        // Is there a more efficient way of doing that?
        VectorXd X_weights_d{X_weights.cast<double>()};
        X_weights_d.array() /= X_weights_d.sum();
        VectorXd Y_weights_d{Y_weights.cast<double>()};
        Y_weights_d.array() /= Y_weights_d.sum();

        return entropicWasserstein<k>(X_centroids, Y_centroids, precision / 3, X_weights_d, Y_weights_d);
    }

} // namespace greenkhorn
#endif /* GREENKHORN */
