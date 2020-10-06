// #define NDEBUG
#define EIGEN_USE_BLAS
#define NDEBUGCERR

#include "utils/samplers.hpp"
#include "utils/vose_alias.hpp"
#include "utils/greenkhorn.hpp"
#include "utils/clustering.hpp"

#include <iostream>
#include <chrono>
#include <vector>
#include <Eigen/Dense>

class Chrono
{
    using clock = std::chrono::high_resolution_clock;

private:
    clock::time_point m_start{clock::now()};

    template <typename T>
    auto nano_cast(T delta_t)
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(delta_t).count();
    }

public:
    void reset()
    {
        m_start = clock::now();
    }

    auto elapsed()
    {
        return nano_cast(clock::now() - m_start);
    }
};

using Eigen::MatrixXd;
using Eigen::VectorXd;


int main()
{
    Chrono chrono{};
    std::cout << "Test" << std::endl;
    // Test greenkhorn
    using greenkhorn::KHORN_TYPE;
    MatrixXd X{samplers::gaussianMixture(5, 2, 5, 1e-4)};
    MatrixXd Y{samplers::gaussianMixture(3, 2, 5, 1e-4)};

    double eps{1.};
    double eta{1.};
    // std::cout << "Enter desired regularization: " << std::endl;
    // std::cin >> eta;
    // std::cout << "Regularization is: " << eta << std::endl;

    VectorXd u {VectorXd::Zero(X.rows())};
    VectorXd v {VectorXd::Zero(Y.rows())};

    chrono.reset();
    auto result {greenkhorn::sinkhorn(greenkhorn::pairwiseSquaredDistance(X, Y), 
                                      VectorXd::Ones(X.rows())/X.rows(), VectorXd::Ones(Y.rows())/Y.rows(), eps, eta,
                                      u, v)};
    std::cout << "Classic done in: " << chrono.elapsed() / 1e6 << " ms. Iterations: " << result.errors.size() << ". " << std::endl;
    std::cout << "Error code: " << static_cast<int>(result.k_error) << std::endl;
    std::cout << "Transport cost:\n " << result.transportCost() << std::endl;

#if 0
    // Test KMeanspp
    chrono.reset();
    MatrixXd m{MatrixXd::Random(10000, 2)};
    auto result {clustering::kmeanspp(m, 100)};
    std::cout << "In : " << chrono.elapsed()/1e6 << "ms." << std::endl;
    // std::cout << result.transpose() << std::endl;
    
    // Measure speed of rounding trick
    Eigen::Index size {10};
    MatrixXd m(size, size);
    VectorXd r(size);
    VectorXd c(size);

    size_t n_trials {5};
    std::vector<double> result(n_trials);
    for (size_t i = 0; i < n_trials; i++)
    {   
        std::cout << "Run " << i;
        m = Eigen::MatrixXd::Random(size, size);
        r = Eigen::VectorXd::Random(size);
        c = Eigen::VectorXd::Random(size);
        r /= r.sum();
        c /= c.sum();
        chrono.reset();
        greenkhorn::roundingTrick(m, r, c);
        result[i] = chrono.elapsed() / 1e6;
        std::cout << " (t = " << result[i] << " ms)" << std::endl;
    }

    std::cout << "Mean over " << n_trials << " runs: \n"
              << std::accumulate(result.begin(), result.end(), 0.) / n_trials << std::endl;

    // Measure speed of AFK-MC^2
    MatrixXd m {MatrixXd::Random(1000000, 3)};

    chrono.reset();
    auto result{afkmc2(m, 1000, 1000)};
    std::cout << chrono.elapsed() / 1e6 << std::endl;

    for (auto val : result)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Measure speed of Vose Sampling
    std::vector<double> proba{.5, .3, .15, .05};
    VoseAlias<double> sampler{&proba[0], 4};
    auto values_2{sampler.sample(1000)};
    mean_count(values_2, 4);

    size_t size{100000};

    chrono.reset();
    std::vector<long int> test;
    test.reserve(size);
    for (size_t i = 0; i < size; i++)
    {
        test.push_back(random());
    }
    std::cout << chrono.elapsed() << std::endl;
    std::cout << test.capacity() << " " << test.size() << std::endl;

    chrono.reset();
    std::vector<long int> test_std_2(size);
    for (auto &val : test_std_2)
    {
        val = random();
    }
    std::cout << chrono.elapsed() << std::endl;

    chrono.reset();
    long int *test2{new long int[size]{}};
    for (size_t i = 0; i < size; i++)
    {
        test2[i] = random();
    }
    std::cout << chrono.elapsed() << std::endl;

#endif
    return 0;
}