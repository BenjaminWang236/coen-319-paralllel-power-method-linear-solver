#include <iostream>
#include <Eigen/Dense>

typedef Eigen::Matrix<long double, Eigen::Dynamic, 1> Vec;
typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> Mat;
// typedef Matrix<std::complex<double>, 2, 2> Eigen::Matrix2cd;
// typedef Matrix<std::complex<float>, 2, 2> Eigen::Matrix2cf;
// typedef Matrix<double, 2, 2> Eigen::Matrix2d;
// typedef Matrix<float, 2, 2> Eigen::Matrix2f;
// typedef Matrix<int, 2, 2> Eigen::Matrix2i;
// typedef Matrix<std::complex<double>, 2, Dynamic> Eigen::Matrix2Xcd;
// typedef Matrix<std::complex<float>, 2, Dynamic> Eigen::Matrix2Xcf;
// typedef Matrix<double, 2, Dynamic> Eigen::Matrix2Xd;
// typedef Matrix<float, 2, Dynamic> Eigen::Matrix2Xf;
// typedef Matrix<int, 2, Dynamic> Eigen::Matrix2Xi;
// typedef Matrix<std::complex<double>, 3, 3> Eigen::Matrix3cd;

using namespace std;

Mat matrix_power_naive(Mat A, int x)
{
    Mat result;
    return result;
}

/** 
 * @brief Power Iteration Serial
 * 
 * Referencing the Pseudocode in tutorial:
 * @cite Mike Koltsov 
 * @link https://www.hackerearth.com/practice/notes/matrix-exponentiation-1/ @endlink
 */
int main()
{
    cout << "Serial Version of Power Iteration/Method Linear Solver" << endl;

    Mat(2, 2) A;
    A << 1, 2, 3, 4;
    cout << "Matrix A sizes is rows: " << A.rows() << " and columns: " << A.cols() << endl;
    matrix_power_naive(A, 2);

    return 0;
}