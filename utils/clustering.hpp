#ifndef UTILS_CLUSTERING
#define UTILS_CLUSTERING

// #define EIGEN_RUNTIME_NO_MALLOC // to allow or not the allocation of temporary variables
// Use it with: Eigen::internal::set_is_malloc_allowed(false);
#include <Eigen/Dense>
#include <Eigen/Core>
#include <random>
#include <map>

#include "vose_alias.hpp"

// Use row order storage layout and put the points in rows
// Use reductions on columns and the right norm computation to do everything in a row

namespace clustering
{
    // using Eigen::MatrixXd;
    using Eigen::VectorXd;
    using Eigen::VectorXi;
    using Eigen::Ref;
    using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using index_t = Eigen::Index;
    using VectorXind_t = Eigen::Matrix<index_t, Eigen::Dynamic, 1>;

    std::vector<index_t> afkmc2(Ref<MatrixXdR> &points, int k, int m)
    {
        // TODO: Change return type to VectorXind_t
        // TODO: Change the points matrix to have coordinates in rows, for better performances.
        // Or use a rowMajor points matrix.

        // Can put all that into a class
        static std::random_device rd_{};
        static std::mt19937 rand_gen{rd_()};
        std::uniform_int_distribution<int> discrete_distribution(0, points.rows() - 1);
        static std::uniform_real_distribution<double> m_uniform_distribution{0., 1.};

        // Sample the first element
        std::vector<index_t> result;
        result.reserve(static_cast<std::size_t>(k));
        result.push_back(discrete_distribution(rand_gen));

        // Define another array containing the cluster centers. This could be avoided, but enable
        // broadcasting with Eigen. In Version 3.3.9 of Eigen, we will be able to take slices and remove
        // this variable.
        MatrixXdR clusters(k, points.cols());
        clusters.row(0) = points.row(result[0]);

        VectorXd distance_to_first(points.cols());
        distance_to_first = (points.rowwise() - points.row(result[0])).rowwise().squaredNorm();
        distance_to_first.array() /= 2 * distance_to_first.sum();
        distance_to_first.array() += 1. / (2 * points.rows());

        // Define Vose Sampler
        VoseAlias<double> sampler{distance_to_first.data(), static_cast<int>(points.rows())};

        index_t x_id;
        index_t y_id;
        double x_distance;
        double y_distance;

        for (int i = 1; i < k; i++)
        {
            x_id = static_cast<index_t>(sampler.sample()); // The sampler returns unsigned int whereas Eigen uses long int
            x_distance = (clusters.topRows(i).rowwise() - points.row(x_id)).rowwise().squaredNorm().minCoeff();
            for (int j = 1; j < m; j++)
            {
                y_id = static_cast<index_t>(sampler.sample());
                y_distance = (clusters.topRows(i).rowwise() - points.row(y_id)).rowwise().squaredNorm().minCoeff();

                if ((x_distance * distance_to_first(y_id) < std::numeric_limits<double>::epsilon()) ||
                    ((y_distance * distance_to_first(x_id)) / (x_distance * distance_to_first(y_id)) > m_uniform_distribution(rand_gen)))
                {
                    x_id = y_id;
                    x_distance = y_distance;
                }
            }

            result.push_back(x_id);
            clusters.row(i) = points.row(x_id);
        }

        return result;
    }

    VectorXind_t afkmc2Eig(Ref<MatrixXdR> &points, int k, int m)
    {
        auto raw {afkmc2(points, k, m)};
        Eigen::Map< VectorXind_t > result{raw.data(), static_cast<Eigen::Index>(raw.size())};
        return result;
    }

    inline double addPoint(const MatrixXdR &points, const VectorXind_t &candidates,
                           VectorXd &distances, VectorXind_t &assignment,
                           VectorXind_t &assignment_candidate_best, VectorXind_t &assignment_candidate_other,
                           VectorXd &distances_candidate_best, VectorXd &distances_candidate_other);

    VectorXind_t kmeanspp(const Ref<MatrixXdR> &points, const int k, const double eps = 0.)
    {
        // TODO: Check well functionning

        // Using VectorXind_t instead of std::vector to avoid conversion from size_type to index_t when indexing
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> discrete_dist(0, points.rows() - 1);

        const size_t n_candidates = static_cast<size_t>(k > 0 ? 2 + int(log(k + 1)) : 8);

        // Draw first centroid randomly
        VectorXind_t assignment(points.rows());
        index_t first_centroid{discrete_dist(gen)};
        for (index_t i = 0; i < points.rows(); i++)
        {
            assignment[i] = first_centroid;
        }

        VectorXd distances{
            (points.rowwise() - points.row(assignment[0])).rowwise().squaredNorm()};

        // Candidates at each round
        VectorXind_t candidates(n_candidates);

        // Memory allocated to store the information relative to the candidates
        VectorXind_t assignment_candidate_best(points.rows());
        VectorXind_t assignment_candidate_other(points.rows());
        VectorXd distances_candidate_best(points.rows());
        VectorXd distances_candidate_other(points.rows());

        double inertia{distances.sum()};
        int n_centroids{1};

        while ((n_centroids < k) && (inertia > eps))
        {
            for (index_t j = 0; j < candidates.rows(); j++)
                candidates[j] = discrete_dist(gen);
            inertia = addPoint(points, candidates, distances, assignment, assignment_candidate_best, assignment_candidate_other,
                               distances_candidate_best, distances_candidate_other);
            n_centroids ++;
        }

        return assignment;
    }

    std::pair<VectorXind_t, VectorXi> assignmentToCount(const VectorXind_t &assignment)
    {
        // Use a map to count the occurence 
        std::map<index_t, int> accumulator;
        for (index_t i = 0; i < assignment.rows(); i++)
        {
            accumulator[assignment[i]] ++;
        }

        VectorXind_t centroid_ids(accumulator.size());
        VectorXi centroid_weights(accumulator.size());

        index_t i {0};
        for (const auto &[key, val]: accumulator)
        {
            centroid_ids[i] = key;
            centroid_weights[i] = val;
            i ++;
        }

        return std::pair<VectorXind_t, VectorXi>(centroid_ids, centroid_weights);
    }

    /**
     * @brief Returns the best candidate among a list of candidate, based on inertia.
     * Arrays containing the assignment and the distance are passed by reference rather than allocated inplace, to prevent multiple allocation.
     * 
     * @param points Matrix containing the points in rows. TODO: put the points in columns for better performance. Or pass a RowMajor Matrix. 
     * @param candidates Vector of Eigen::Index containing the indices of the candidates. The next centroid will be chosen among those.
     * @param distances Current distances of the points to the current centroids. Will be updated with the best candidate. 
     * @param assignment Current assignment of the points to the current centroids. Will be updated with the best candidate.
     * @param assignment_candidate_best Memory allocated to temporarily store the assignment relative to the best candidate. Modified. Not to be relied upon.
     * @param assignment_candidate_other Idem.
     * @param distances_candidate_best Idem.
     * @param distances_candidate_other Idem.
     */
    inline double addPoint(const MatrixXdR &points, const VectorXind_t &candidates,
                           VectorXd &distances, VectorXind_t &assignment,
                           VectorXind_t &assignment_candidate_best, VectorXind_t &assignment_candidate_other,
                           VectorXd &distances_candidate_best, VectorXd &distances_candidate_other)
    {
        index_t id_candidate_best{candidates[0]};
        double inertia_candidate_best{0.};
        double inertia_candidate_other{0.};

        // Compute inertia of the first candidate
        // This could be done with array indexing in Eigen 3.3.9, with, e.g:
        // ... distances_candidate_best = (points.rowwise() - points.row(id_candidate_best)).rowwise().squaredNorm();
        // ... distances_candidate_best = distances_candidate_best.cwiseMin(distances);
        for (index_t i = 0; i < points.rows(); i++)
        {
            distances_candidate_best[i] = (points.row(i) - points.row(id_candidate_best)).squaredNorm();
            if (distances[i] < distances_candidate_best[i])
            {
                distances_candidate_best[i] = distances[i];
                assignment_candidate_best[i] = assignment[i];
            }
            else
            {
                assignment_candidate_best[i] = id_candidate_best;
            }
        }
        inertia_candidate_best = distances_candidate_best.sum();

        for (index_t j = 1; j < candidates.size(); j++)
        {
            for (index_t i = 0; i < points.rows(); i++)
            {
                distances_candidate_other[i] = (points.row(i) - points.row(candidates[j])).squaredNorm();
                if (distances[i] < distances_candidate_other[i])
                {
                    distances_candidate_other[i] = distances[i];
                    assignment_candidate_other[i] = assignment[i];
                }
                else
                {
                    assignment_candidate_other[i] = candidates[j];
                }
            }
            inertia_candidate_other = distances_candidate_other.sum();

            // Should we keep this new candidate or the former?
            if (inertia_candidate_other < inertia_candidate_best)
            {
                inertia_candidate_best = inertia_candidate_other;
                assignment_candidate_best = assignment_candidate_other;
                distances_candidate_best = distances_candidate_other;
                id_candidate_best = candidates[j];
            }
        }

        // Perform the change
        distances = distances_candidate_best;
        assignment = assignment_candidate_best;
        return inertia_candidate_best;
    }
} // namespace clustering
#endif /* UTILS_CLUSTERING */
