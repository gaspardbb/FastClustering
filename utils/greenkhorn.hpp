#ifndef GREENKHORN
#define GREENKHORN

#include <Eigen/Dense>
#include <vector>
#include "utils/clustering.hpp"
#include <math.h> // assert, isnan

#ifdef DEBUGCERR
#include <iostream> // debug
#endif

namespace greenkhorn
{

    using Eigen::MatrixXd;
    using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Eigen::Ref;
    using Eigen::VectorXd;

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
        HASNAN,
    };

    std::string khornErrorMessage(KHORN_ERROR k)
    {
        switch (k)
        {
        case KHORN_ERROR::SUCCESS:
            return std::string("0: Khorn converged successfully.");
        case KHORN_ERROR::ITER_MAX:
            return std::string("1: Maximum number of iterations reached. Precision is probably below machine's precision.");
        case KHORN_ERROR::HASNAN:
            return std::string("2: Floating point errors, the result contains nan values.");
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
        double eta{0.};
        double eps{0.};
        double eps_prime{0.};

        // Enable to set the duality gap before it is rounded
        double dualityGap(const VectorXd &r, const VectorXd &c)
        {
            return (
                // primal min:
                (cost_matrix.array() * transport_plan.array()).sum() + eta * (transport_plan.array() * transport_plan.array().log()).sum()
                // dual max
                - eta * (u.dot(r) + v.dot(c) + 1 - transport_plan.sum()));
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

    inline void logSumRows(const MatrixXdR &log_M, VectorXd &log_out)
    {
        VectorXd max_shift{log_M.rowwise().maxCoeff()};
        log_out = max_shift.array() + (log_M.colwise() - max_shift).array().exp().rowwise().sum().log();
    }

    inline void logSumCols(const MatrixXdR &log_M, VectorXd &log_out)
    {
        Eigen::RowVectorXd max_shift{log_M.colwise().maxCoeff()};
        log_out = (max_shift.array() + (log_M.rowwise() - max_shift).array().exp().colwise().sum().log()).transpose();
    }

    inline void logSumVecs(const VectorXd &x, const VectorXd &y, VectorXd &out)
    {
#pragma omp parallel for
        for (Eigen::Index i = 0; i < x.rows(); i++)
        {
            out(i) = (x(i) > y(i)) ? x(i) + log(1 + exp(y(i) - x(i))) : y(i) + log(1 + exp(x(i) - y(i)));
        }
    }

    inline void logDiffVecs(const VectorXd &x, const VectorXd &y, VectorXd &out)
    {
        out.array() = x.array() + (1 - (y - x).array().exp()).log();
    }

    inline double constraintsViolation(const VectorXd &log_sum_rows, const VectorXd &r, const VectorXd &log_sum_cols, const VectorXd &c)
    {
        return (log_sum_rows.array().exp() - r.array()).matrix().lpNorm<1>() + (log_sum_cols.array().exp() - c.array()).matrix().lpNorm<1>();
    }

    khorn_return greenkhorn(const MatrixXdR &C, const VectorXd &r, const VectorXd &c, double eps, double eta,
                            VectorXd &f, VectorXd &g, int iter_max = 100000, int max_recovered_nan = 1)
    {
        // const values to speed up computation, at the price of some memory
        const VectorXd log_r{r.array().log().matrix()};
        const VectorXd log_c{c.array().log().matrix()};
        const VectorXd r_log_prim{-r + (r.array() * log_r.array()).matrix()};
        const VectorXd c_log_prim{-c + (c.array() * log_c.array()).matrix()};

        // const auto N{
        //     std::max(C.rows(), C.cols())};
        // const double R{
        //     C.maxCoeff() / eta + log(N) - 2 * log(std::min(r.minCoeff(), c.minCoeff()))};
        // const int iter_max{static_cast<int>(2. + 112. * N * R / eps)};

        MatrixXdR log_B_t{C / eta}; // coupling matrix
        log_B_t.array() -= log_B_t.maxCoeff();
        log_B_t.colwise() += f;             // scale rows
        log_B_t.rowwise() += g.transpose(); // scale cols
        VectorXd log_r_B{log_B_t.rows()};   // sum over rows
        logSumRows(log_B_t, log_r_B);
        VectorXd log_c_B{log_B_t.cols()}; // sum over cols
        logSumCols(log_B_t, log_c_B);

#ifdef DEBUGCERR
        std::cerr << "+++++ INITIALIZATION +++++" << std::endl;
        std::cerr << "log B_t is: \n"
                  << log_B_t << "\n and B_t is: \n"
                  << log_B_t.array().exp() << std::endl;
        std::cerr << "log r_B is: \n"
                  << log_r_B.transpose() << "\n and r_B is: \n"
                  << log_r_B.array().exp().transpose() << std::endl;
        std::cerr << "log c_B is: \n"
                  << log_c_B.transpose() << "\n and c_B is: \n"
                  << log_c_B.array().exp().transpose() << std::endl;
#endif

        VectorXd::Index r_ind{}; // Max gain over rows and cols,
        VectorXd::Index c_ind{};
        double r_value{}; // And associated value
        double c_value{};

        VectorXd r_gain{log_r_B.array().exp() - r.array() * log_r_B.array() + r_log_prim.array()};
        VectorXd c_gain{log_c_B.array().exp() - c.array() * log_c_B.array() + c_log_prim.array()};

        // Values updated in the loop
        double E_t{constraintsViolation(log_r_B, r, log_c_B, c)};
        int n_iter{0};

        std::vector<double> errors;
        errors.push_back(E_t);

        int recovered_nan{0}; // Number of times we had to recover the nan values;

        while ((n_iter < iter_max) && (E_t > eps))
        {
#ifdef DEBUGCERR
            std::cerr << "===== ITER " << n_iter << " =====" << std::endl;
            std::cerr << "r gain:  " << r_gain.transpose()
                      << "\ncomputed:" << (log_r_B.array().exp() - r.array() * log_r_B.array() + r_log_prim.array()).transpose()
                      << "\nc_gain:  " << c_gain.transpose()
                      << "\ncomputed:" << (log_c_B.array().exp() - c.array() * log_c_B.array() + c_log_prim.array()).transpose()
                      << std::endl;
            std::cerr << "Coupling Matrix: \n"
                      << log_B_t << "\n"
                      << std::endl;
#endif

            r_value = r_gain.maxCoeff(&r_ind);
            c_value = c_gain.maxCoeff(&c_ind);

            if (r_value > c_value)
            { // Update rows
                /* Previously I did:
                log_scale = log_r(r_ind) - log_r_B(r_ind);
                log_c_B = (log_c_B.array().exp() + (1 - exp(-log_scale)) * log_B_t.row(r_ind).array().exp().transpose()).log();

                It requires less computation, but it can't be done in the log space, as the r.h. term can be negative.
                The sum is, by definition, positive.
              */
                logDiffVecs(log_c_B, log_B_t.row(r_ind).transpose(), log_c_B);

#ifdef DEBUGCERR
                std::cerr << "\tLog_cB after substracting:\n\t" << log_c_B.transpose() << std::endl;
#endif

                f(r_ind) += log_r(r_ind) - log_r_B(r_ind);                   // Update dual variable
                log_B_t.row(r_ind).array() += log_r(r_ind) - log_r_B(r_ind); // Update coupling matrix

                log_r_B(r_ind) = log_r(r_ind);                                // Update sum over row
                logSumVecs(log_c_B, log_B_t.row(r_ind).transpose(), log_c_B); // Update sum over col, in exp domain
                r_gain(r_ind) = 0.;
                c_gain = log_c_B.array().exp() - c.array() * log_c_B.array() + c_log_prim.array();
            }
            else
            { // Update cols
                logDiffVecs(log_r_B, log_B_t.col(c_ind), log_r_B);

#ifdef DEBUGCERR
                std::cerr << "\tLog_rB after substracting:\n\t" << log_r_B.transpose() << std::endl;
#endif

                g(c_ind) += log_c(c_ind) - log_c_B(c_ind);
                log_B_t.col(c_ind).array() += log_c(c_ind) - log_c_B(c_ind);

                log_c_B(c_ind) = log_c(c_ind);
                // log_r_B = (log_r_B.array().exp() + (1 - exp(-log_scale)) * log_B_t.col(c_ind).array().exp()).log();
                logSumVecs(log_r_B, log_B_t.col(c_ind), log_r_B);
                c_gain(c_ind) = 0.;
                r_gain = log_r_B.array().exp() - r.array() * log_r_B.array() + r_log_prim.array();
            }

            E_t = constraintsViolation(log_r_B, r, log_c_B, c);

            if (!std::isnormal(E_t) && (recovered_nan < max_recovered_nan))
            {
                // There were numerical errors, and we allow trying to recover from those
                // Those are O(n^2) computations
#ifdef DEBUGCERR
                std::cerr << "\tGot E_t: " << E_t << " and attempting to recover for the" << recovered_nan
                          << " times. (" << max_recovered_nan << " allowed).\n";
#endif
                logSumRows(log_B_t, log_r_B);
                logSumCols(log_B_t, log_c_B);
                E_t = constraintsViolation(log_r_B, r, log_c_B, c);
                r_gain = log_r_B.array().exp() - r.array() * log_r_B.array() + r_log_prim.array();
                c_gain = log_c_B.array().exp() - c.array() * log_c_B.array() + c_log_prim.array();
#ifdef DEBUGCERR
                std::cerr << "\tnew log_r_B: " << log_r_B.transpose() << "\n";
                std::cerr << "\tnew log_c_B: " << log_c_B.transpose() << "\n";
                std::cerr << "\tnew E_t: " << E_t << std::endl;
#endif
            }

#ifdef DEBUGCERR

            std::cerr << ((r_value > c_value) ? "f" : "g") << "changed !" << std::endl;
            std::cerr << "new f: \n"
                      << f.transpose() << std::endl;
            std::cerr << "new g: \n"
                      << g.transpose() << std::endl;
            std::cerr << "E_t: " << E_t << "\n"
                      << std::endl;

            VectorXd debug_logsum_row{log_r_B.rows()};
            logSumRows(log_B_t, debug_logsum_row);
            VectorXd debug_logsum_col{log_c_B.rows()};
            logSumCols(log_B_t, debug_logsum_col);

            std::cerr << "log_r_B: \n"
                      << log_r_B.transpose() << std::endl;
            std::cerr << "Computed sum of rows: \n"
                      << debug_logsum_row.transpose() << std::endl;
            std::cerr << "log_c_B: \n"
                      << log_c_B.transpose() << std::endl;
            std::cerr << "Computed sum of cols: \n"
                      << debug_logsum_col.transpose() << std::endl;
#endif

            errors.push_back(E_t);
            ++n_iter;
        }

        KHORN_ERROR error_code;
        if (n_iter == iter_max)
            error_code = KHORN_ERROR::ITER_MAX;
        else if (!std::isnormal(E_t))
            error_code = KHORN_ERROR::HASNAN;
        else
            error_code = KHORN_ERROR::SUCCESS;

        errors.shrink_to_fit(); // return unused allocated memory to the OS
        khorn_return result{f, g, errors, log_B_t.array().exp(), C, error_code, KHORN_TYPE::GREEN};
        result.duality_gap = result.dualityGap(r, c);
        return result;
    }

    khorn_return sinkhorn(const MatrixXdR &C, const VectorXd &r, const VectorXd &c, double eps, double eta,
                          VectorXd &f, VectorXd &g, int iter_max = 100000)
    {
        // const double R{
        // -log((-C.array() / eta).exp().minCoeff() * std::min(r.minCoeff(), c.minCoeff()))};
        // const int iter_max{static_cast<int>(2. + 4. * R / eps)};

        // const values to speed up computation, at the price of some memory
        const VectorXd log_r{r.array().log().matrix()};
        const VectorXd log_c{c.array().log().matrix()};

        // Values updated in the loop
        MatrixXdR log_B_t{C / eta};
        log_B_t.array() -= log_B_t.maxCoeff();
        log_B_t.colwise() += f;             // scale rows
        log_B_t.rowwise() += g.transpose(); // scale cols
        VectorXd log_r_B{C.rows()};         // Compute sum over rows
        logSumRows(log_B_t, log_r_B);       // ...
        VectorXd log_c_B{C.cols()};         // and sum over columns
        logSumCols(log_B_t, log_c_B);
        VectorXd log_scale_rows{C.rows()};
        VectorXd log_scale_cols{C.cols()};

        double E_t{constraintsViolation(log_r_B, r, log_c_B, c)};
        int n_iter{0};
        std::vector<double> errors;
        errors.push_back(E_t);

        while ((n_iter < iter_max) && E_t > eps)
        {
            log_scale_rows = log_r.array() - log_r_B.array();
            f += log_scale_rows;
            log_B_t.colwise() += log_scale_rows;
            assert(!log_B_t.hasNaN());
            logSumCols(log_B_t, log_c_B);

            log_scale_cols = log_c.array() - log_c_B.array();
            g += log_scale_cols;
            log_B_t.rowwise() += log_scale_cols.transpose();
            assert(!log_B_t.hasNaN());
            logSumRows(log_B_t, log_r_B);

            E_t = constraintsViolation(log_r_B, r, log_c_B, c);

            errors.push_back(E_t);
            ++n_iter;
        }

        KHORN_ERROR error_code;
        if (n_iter == iter_max)
            error_code = KHORN_ERROR::ITER_MAX;
        else if (!std::isnormal(E_t))
            error_code = KHORN_ERROR::HASNAN;
        else
            error_code = KHORN_ERROR::SUCCESS;

        errors.shrink_to_fit(); // return unused allocated memory to the OS

        khorn_return result{f, g, errors, log_B_t.array().exp(), C, error_code, KHORN_TYPE::SIN};
        result.duality_gap = result.dualityGap(r, c);
        return result;
    }

    template <KHORN_TYPE k>
    khorn_return khorn(const MatrixXdR &C, const VectorXd &r, const VectorXd &c, double eps, double eta,
                       VectorXd &initial_u, VectorXd &initial_v, int iter_max = 10000)
    {
        // TODO: Factorize these parts
        switch (k)
        {
        case KHORN_TYPE::GREEN:
            return greenkhorn(C, r, c, eps, eta, initial_u, initial_v, iter_max);
            break;

        case KHORN_TYPE::SIN:
            return sinkhorn(C, r, c, eps, eta, initial_u, initial_v, iter_max);
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
                       VectorXd &initial_u, VectorXd &initial_v, int iter_max = 10000)
    {
        // TODO: check marginals are in the probability simplex
        const auto n{
            std::min(C.rows(), C.cols())};
        const double eta{
            eps / (4 * log(n + 1))};
        const double eps_prime{
            eps / (8 * C.maxCoeff())};

        khorn_return result = khorn<k>(C,
                                       (1 - eps_prime / 8) * r.array() + (eps_prime / (8 * C.rows())),
                                       (1 - eps_prime / 8) * c.array() + (eps_prime / (8 * C.cols())),
                                       eps_prime / 2, eta,
                                       initial_u, initial_v, iter_max);
        roundingTrick(result.transport_plan, r, c);
        result.eps = eps;
        result.eps_prime = eps_prime;
        result.eta = eta;
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
                                     VectorXd &initial_u, VectorXd &initial_v, int iter_max = 10000)
    {
        MatrixXdR C{pairwiseSquaredDistance(X, Y)};
        return khorn<k>(C, r.array() / r.sum(), c.array() / c.sum(), precision, initial_u, initial_v, iter_max);
    }

    template <KHORN_TYPE k>
    khorn_return entropicWasserstein(const MatrixXdR &X, const MatrixXdR &Y,
                                     const VectorXd &r, const VectorXd &c,
                                     double precision, int iter_max=100000)
    {
        VectorXd initial_u{VectorXd::Zero(X.rows())};
        VectorXd initial_v{VectorXd::Zero(Y.rows())};
        return entropicWasserstein<k>(X, Y, r, c, precision, initial_u, initial_v, iter_max);
    }

    template <KHORN_TYPE k>
    khorn_return entropicWassersteinWarmStart(const Ref<const MatrixXdR> &X, const Ref<const MatrixXdR> &Y, double precision,
                                              VectorXd &initial_u, VectorXd &initial_v, int iter_max)
    {
        return entropicWasserstein<k>(X, Y, VectorXd::Ones(X.rows()), VectorXd::Ones(Y.rows()), precision, initial_u, initial_v, iter_max);
    }

    template <KHORN_TYPE k>
    khorn_return entropicWasserstein(const Ref<const MatrixXdR> &X, const Ref<const MatrixXdR> &Y, double precision, int iter_max = 10000)
    {
        VectorXd initial_u{VectorXd::Zero(X.rows())};
        VectorXd initial_v{VectorXd::Zero(Y.rows())};
        return entropicWassersteinWarmStart<k>(X, Y, precision, initial_u, initial_v, iter_max);
    }

    template <KHORN_TYPE k>
    khorn_return entropicWassersteinAddPoint(const Ref<const MatrixXdR> &C, const Ref<const MatrixXdR> &X, const Ref<const MatrixXdR> &Y,
                                             const Ref<const VectorXd> &x, const Ref<const VectorXd> &y, double precision,
                                             const Ref<const VectorXd> &initial_u, const Ref<const VectorXd> &initial_v)
    {
        // This is probably not the most efficient way to recopy the whole matrix; just for test purpose for now
        // Probably easier to create a class with its own X, Y, C, r, c, and so on.
        // TODO: clean this if used more
        MatrixXd new_C{C.rows() + 1, C.cols() + 1};
        new_C.topLeftCorner(C.rows(), C.cols()) = C;
        new_C.bottomLeftCorner(1, C.rows()) = pairwiseSquaredDistance(X, y.transpose()).transpose();
        new_C.topRightCorner(C.cols(), 1) = pairwiseSquaredDistance(Y, x.transpose());
        new_C(C.rows(), C.cols()) = (x - y).squaredNorm();

        VectorXd u{VectorXd::Zero(initial_u.rows() + 1)}, v{VectorXd::Zero(initial_v.rows() + 1)};
        u.head(initial_u.rows()) = initial_u;
        v.head(initial_v.rows()) = initial_v;

        auto r{VectorXd::Ones(X.rows() + 1).array()};
        auto c{VectorXd::Ones(Y.rows() + 1).array()};

        return khorn<k>(new_C, r.array() / r.sum(), c.array() / c.sum(), precision);
    }

    template <KHORN_TYPE k>
    khorn_return entropicWassersteinKMeans(const Ref<const MatrixXdR> &X, const Ref<const MatrixXdR> &Y, double precision,
                                           int iter_max = 100000)
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

        return entropicWasserstein<k>(X_centroids, Y_centroids, X_weights_d, Y_weights_d, precision / 3, iter_max);
    }

    class PairedOnlineKMeans
    {
    public:
        // TODO: Prefer pointers?
        clustering::OnlineKMeans centroids_x;
        clustering::OnlineKMeans centroids_y;

    private:
        MatrixXd cost;
        Eigen::Index n_tot;

    public:
        // TODO: pass by reference? Or not that important?
        PairedOnlineKMeans(clustering::OnlineKMeans c_x, clustering::OnlineKMeans c_y) : centroids_x{c_x},
                                                                                         centroids_y{c_y},
                                                                                         cost{pairwiseSquaredDistance(centroids_x.getCentroids(), centroids_y.getCentroids())},
                                                                                         n_tot{c_x.getNTot()}
        {
        }

        PairedOnlineKMeans(MatrixXdR &initial_x, MatrixXdR &initial_y, double learning_rate) : PairedOnlineKMeans(clustering::OnlineKMeans(initial_x, learning_rate),
                                                                                                                  clustering::OnlineKMeans(initial_y, learning_rate))
        {
        }

        void addPoints(const VectorXd &x, const VectorXd &y)
        {
            // Update centroids
            Eigen::Index id_x{centroids_x.addPoint(x)};
            Eigen::Index id_y{centroids_y.addPoint(y)};

            // Update cost matrix
            cost.row(id_x) = pairwiseSquaredDistance(centroids_x.getCentroids().row(id_x),
                                                     centroids_y.getCentroids());
            cost.col(id_y) = pairwiseSquaredDistance(centroids_x.getCentroids(),
                                                     centroids_y.getCentroids().row(id_y));

            n_tot += 1;
        }

        void addPointsSoft(const VectorXd &x, const VectorXd &y)
        {
            // Update centroids
            centroids_x.addPointSoft(x);
            centroids_y.addPointSoft(y);

            // Update cost matrix: a slightly more clever update exists, still in O(k^2) though
            // TODO: Check this alternative method
            cost = pairwiseSquaredDistance(centroids_x.getCentroids(), centroids_y.getCentroids());

            n_tot += 1;
        }

        const MatrixXd &getCost() const { return cost; }
    };

} // namespace greenkhorn
#endif /* GREENKHORN */
