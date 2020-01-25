#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <iostream>

int
main()
{
    // clang-format off
    Eigen::MatrixXf v1 = (Eigen::MatrixXf(5, 3) <<
                          -10.0, 0.0, 1.0,  // first
                          5.3, 7.7, 1.0,  // second
                          3.1, 3.2, 10.1,  // third
                          -5.5, 1.3, 3.3,  // fourth
                          -7.1, 6.8, -4.2  // fifth
                          )
                             .finished();

    Eigen::MatrixXf s = (Eigen::MatrixXf(5, 4) <<
                         1.0, -0.2, 0.0, -0.5, // first
                         0.8, 0.7, 1.0, 0.0, // second
                         0.5, 0.2, 0.1, -0.3, // third
                         0.3, 0.3, 0.3, -0.2, // fourth
                         0.1, 0.8, -0.2, -1.0  // fifth
                         )
                            .finished();

    s.col(3) *= 0.01;
    // clang-format on

    std::cout << "v1\n" << v1 << std::endl;

    Eigen::SparseMatrix<float> A(3 * v1.rows(), 3 * s.cols());
    A.reserve(Eigen::VectorXi::Constant(3 * s.cols(), v1.rows()));
    for (long i = 0, n = v1.rows(); i < n; ++i)
    {
        long i3   = i * 3;
        long i3_1 = i3 + 1;
        long i3_2 = i3 + 2;
        long m    = s.cols();
        long m2   = 2 * m;
        for (long j = 0; j < m; ++j)
        {
            A.insert(i3, j)        = s(i, j);
            A.insert(i3_1, m + j)  = s(i, j);
            A.insert(i3_2, m2 + j) = s(i, j);
        }
    }
    // std::cout << "A\n" << A << std::endl;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> v1_rm = v1;
    Eigen::VectorXf B = Eigen::Map<Eigen::VectorXf>(v1_rm.data(), 3 * v1.rows());

    // std::cout << "B\n" << B << std::endl;

    Eigen::SparseMatrix<float, Eigen::RowMajor> ATA = A.transpose() * A;
    Eigen::MatrixXf                             ATB = A.transpose() * B;

    ATA.diagonal() += Eigen::VectorXf::Constant(3 * s.cols(), 10.0);

    Eigen::SimplicialLLT<Eigen::SparseMatrix<float>> solver(ATA);
    Eigen::MatrixXf                                  res = solver.solve(ATB);

    Eigen::MatrixXf res_shape = Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        res.data(), 3, s.cols());
    std::cout << "res shape\n" << res_shape << std::endl;

    for (int i = 0; i < 5; ++i)
    {
        Eigen::VectorXf res_v = res_shape * s.row(i).transpose();
        std::cout << "rest v\n " << res_v.transpose() << std::endl;
    }

    return 0;
}
