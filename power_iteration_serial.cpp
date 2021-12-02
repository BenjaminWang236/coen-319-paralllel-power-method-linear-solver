#include <iostream>
#include <stdlib.h>
#include <Eigen/Dense>

/**
 * "echo $CPLUS_INCLUDE_PATH" in Terminal to find path to Eigen
 * Enter the Eigeh path to the includePath of VSCode to resolve the not found error
 * @link
*/

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
using namespace Eigen;

Mat identity_matrix(Mat A)
{
    Mat I(A.rows(), A.rows());
    I = Mat::Zero(A.rows(), A.rows());
    for (int i = 0; i < A.rows(); i++)
    {
        I(i, i) = 1;
    }
    return I;
}

bool isSquareMatrix(Mat A)
{
    return A.rows() == A.cols();
}

void checkMatrixPowerOK(Mat A, int x)
{
    if (!isSquareMatrix(A))
    {
        cout << "ERROR: Matrix is NOT SQUARE!!!" << endl;
        throw exception();
    }
    if (x < 0)
    {
        cout << "ERROR: Exponent must be positive real numbers!" << endl;
        throw exception();
    }
}

Mat matrix_power_naive(Mat A, int x)
{
    checkMatrixPowerOK(A, x);
    Mat result(A.rows(), A.rows());
    result = identity_matrix(A);
    for (int i = 0; i < x; i++)
    {
        result *= A;
    }
    return result;
}

/** 
 * O(n^3 * log_2(x)) time, for n == square matrix A's row/col size
 * For x = 7 == 0x0111, this turns into result = I * A^1 * A^2 * A^4
 * For x = 10 == 0x1010, this turns into result = I * A^2 * A^8
*/
Mat matrix_power_smart(Mat A, int x)
{
    checkMatrixPowerOK(A, x);
    Mat result(A.rows(), A.rows());
    result = identity_matrix(A);
    cout << "x is now:\t";
    while (x > 0)
    {
        cout << x << "\t";
        if (x % 2 == 1)
            result *= A;
        A *= A;
        x /= 2; // Same as x >>= 1
    }
    cout << endl;
    return result;
}

/** 
 * @brief Power Iteration Serial
 * 
 * Referencing the Pseudocode in tutorial:
 * @cite Mike Koltsov 
 * @link https://www.hackerearth.com/practice/notes/matrix-exponentiation-1/ @endlink
 * 
 * Run with "g++ $(pkg-config --cflags eigen3) power_iteration_serial.cpp -o power_iteration_serial && ./power_iteration_serial"
 * @link https://stackoverflow.com/questions/21984971/how-to-compile-a-c-program-using-eigen-without-specifying-the-i-flag @endlink
 */
int main()
{
    try
    {
        cout << "Serial Version of Power Iteration/Method Linear Solver" << endl;
        Mat A(2, 2);
        A << 1, 2, 3, 4;
        cout << "Matrix A sizes is rows: " << A.rows() << " and columns: " << A.cols() << endl;
        Mat B = matrix_power_naive(A, 2);
        cout << "Matrix B: "<< endl << B << endl;
        Mat C = matrix_power_smart(A, 7);
        cout << "Matrix C: " << endl << C << endl;

        // TODO: Now that Matrix Exponentiation is done, implement Matrix-Vector Multiplication, then Power Iterations Algorithm
    }
    catch(exception e)
    {
        return EXIT_FAILURE;
    }
    catch (...)
    {
        cout << "Default Exception" << endl;
        return EXIT_FAILURE;
    }

    return 0;
}