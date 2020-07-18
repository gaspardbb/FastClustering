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