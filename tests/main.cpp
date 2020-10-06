#include "gtest/gtest.h"

#include <Eigen/Dense>
#include "utils/greenkhorn.hpp"

TEST(greenkhorn, logSum)
{
    greenkhorn::MatrixXdR X{greenkhorn::MatrixXdR::Random(10, 3)};
    X.array() += 1.;
    X.array() /= 2.; // X has values between 0 and 2

    Eigen::VectorXd sum_rows{X.rows()};
    Eigen::VectorXd sum_cols{X.cols()};
    greenkhorn::logSumRows(X.array().log(), sum_rows);
    greenkhorn::logSumCols(X.array().log(), sum_cols);

    double row_error{(X.rowwise().sum().array() - sum_rows.array().exp()).abs().sum()};
    double col_error{(X.colwise().sum().array().transpose() - sum_cols.array().exp()).abs().sum()};

    EXPECT_NEAR(row_error, 0., 1e-10) << "Log sum on rows.";
    EXPECT_NEAR(col_error, 0., 1e-10) << "Log sum on cols.";

    Eigen::VectorXd x{Eigen::VectorXd::Random(10)};
    Eigen::VectorXd y{Eigen::VectorXd::Random(10)};
    x.array() += 1.;
    y.array() += 1.;
    Eigen::VectorXd sum_xy{x + y};
    greenkhorn::logSumVecs(x.array().log(), y.array().log(), x);

    double sum_error{(sum_xy.array() - x.array().exp()).abs().sum()};
    EXPECT_NEAR(sum_error, 0., 1e-10) << "Log sum of two vectors.";

    x = Eigen::VectorXd::Random(10).array() + 3.; // [2., 4.]
    y = Eigen::VectorXd::Random(10).array() + 1.; // [0., 2.]
    Eigen::VectorXd diff_xy{x - y};
    greenkhorn::logDiffVecs(x.array().log(), y.array().log(), x);
    double diff_error{(diff_xy.array() - x.array().exp()).abs().sum()};
    EXPECT_NEAR(diff_error, 0., 1e-10) << "Log diff of two vectors.";
}

#include "utils/clustering.hpp"

TEST(clustering, addPoint)
{
    clustering::MatrixXdR points{6, 2};
    points << 0, -1,
        0, 0,
        0, 1,
        0, 3,
        0, 4,
        0, 5;
    clustering::VectorXind_t candidates{2};
    candidates << 0, 4;

    clustering::VectorXd distances{6};
    distances << 1, 0, 1, 3, 4, 5;
    clustering::VectorXind_t assignment{clustering::VectorXind_t::Ones(6)};

    clustering::VectorXind_t assign_cand_best{6}, assign_cand_other{6};
    clustering::VectorXd dist_cand_best{6}, dist_cand_other{6};

    double inertia{clustering::addPoint(points, candidates, distances, assignment,
                                        assign_cand_best, assign_cand_other, dist_cand_best, dist_cand_other)};

    clustering::VectorXind_t expected_assignment{6};
    expected_assignment << 1, 1, 1, 4, 4, 4;
    clustering::VectorXd expected_distance{6};
    expected_distance << 1, 0, 1, 1, 0, 1;

    EXPECT_NEAR(inertia, 4., 1e-10);
    EXPECT_TRUE(assignment.isApprox(expected_assignment));
    EXPECT_TRUE(distances.isApprox(expected_distance, 1e-10));
}

TEST(clustering, assignmentToCount)
{
    clustering::VectorXind_t assignment{8};
    assignment << 1, 0, 1, 1, 2, 0, 2, 1;
    auto [centroid_id, weights] = clustering::assignmentToCount(assignment);

    clustering::VectorXind_t expected_centroid_id {3};
    expected_centroid_id << 0, 1, 2;
    EXPECT_TRUE(centroid_id.isApprox(expected_centroid_id)) << centroid_id.transpose();

    clustering::VectorXi expected_weights {3};
    expected_weights << 2, 4, 2;
    EXPECT_TRUE(weights.isApprox(expected_weights)) << expected_weights.transpose();
}

TEST(clustering, onlineKMeansHard)
{
    // Sample point cloud
    clustering::MatrixXdR X{3, 2};
    X << 0, 0,
         0, 1,
         1, 0;
    // new ponit near point 0
    Eigen::VectorXd point{2};
    point << -1., -2.;

    // Creating the model and adding the point
    clustering::OnlineKMeans model{X, .1};
    model.addPoint(point);

    // Only first row should be updated
    clustering::MatrixXdR expected_result{3, 2};
    expected_result << -.1, -.2,
        0, 1,
        1, 0;

    for (Eigen::Index i = 0; i < 3; i++)
    {
        for (Eigen::Index j = 0; j < 2; j++)
        {
            EXPECT_NEAR(expected_result(i, j), model.getCentroids()(i, j), 1e-10) << "CENTROIDS: Equality failed at index: " << i << ", " << j;
        }
    }

    clustering::VectorXi expected_weights{3};
    expected_weights << 2, 1, 1;

    for (Eigen::Index i = 0; i < 3; i++)
    {
        EXPECT_EQ(expected_weights(i), model.getWeights()(i)) << "WEIGHTS: Equality failed at index: " << i;
    }
}


TEST(clustering, onlineKMeansSoft)
{
    // Not so confident in the broadcasting I did

    // Sample point cloud
    clustering::MatrixXdR X{3, 2};
    X << 0, 0,
         0, 1,
         1, 0;
    // new ponit near point 0
    Eigen::VectorXd point{2};
    point << -1., -2.;

    // Creating the model and adding the point
    clustering::OnlineKMeans model{X, .1};
    Eigen::VectorXd expected_coeff {3};
    expected_coeff << 1./5., 1./10., 1./8.;
    expected_coeff = (expected_coeff.array() * .5/expected_coeff.sum() + 1./6.);
    
    clustering::MatrixXdR expected_result{3, 2};
    for (Eigen::Index i = 0; i < 3; i++)
    {
        expected_result.row(i) = X.row(i) - expected_coeff(i) * (X.row(i) - point.transpose());
    }

    const Eigen::VectorXd coeff{model.addPointSoft(point)};
    EXPECT_TRUE(expected_result.isApprox(model.getCentroids(), 1e-10)) << model.getCentroids() << "\n expected:\n" << expected_result;
    EXPECT_TRUE(expected_coeff.isApprox(coeff, 1e-10)) << coeff.transpose() << "\n expected:\n" << expected_coeff.transpose();
}