#ifndef UTILS_SAMPLERS
#define UTILS_SAMPLERS

#include <Eigen/Dense>
#include <random>
#include <vector>

namespace samplers
{
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    MatrixXd gaussianMixture(size_t n_samples, size_t dim, size_t n_gaussian, double covariance = 1.)
    {
        static std::mt19937 gen{std::random_device{}()};
        static std::uniform_real_distribution<double> uniform{0., 1.};
        static std::normal_distribution<double> normal{};

        // Random weights
        std::vector<double> weights(n_gaussian);
        for (size_t i = 0; i < n_gaussian; i++)
            weights[i] = uniform(gen);

        // Random means
        MatrixXd mean{n_gaussian, dim};
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n_gaussian); i++)
        {
            for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(dim); j++)
            {
                mean(i, j) = uniform(gen);
            }
        }

        std::discrete_distribution<int> discrete(weights.begin(), weights.end());

        MatrixXd result{n_samples, dim};
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(n_samples); i++)
        {
            result.row(i) = mean.row(discrete(gen)) + covariance * Eigen::VectorXd{dim}.unaryExpr(
                                                                       [&](auto x) { (void) x; return normal(gen); }).transpose();
        }

        return result;
    }

} // namespace samplers

#endif /* UTILS_SAMPLERS */
