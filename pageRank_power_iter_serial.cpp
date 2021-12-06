#include <iostream>
#include <stdlib.h>
#include <Eigen/Dense>

#define MAX_ITER 50
#define EPSILON 0.0000001
#define DEBUG_PRINT 0 // 0: no print, 1: print
#define TELEPORT_PARAMETER 0.8

/**
 * "echo $CPLUS_INCLUDE_PATH" in Terminal to find path to Eigen after it's included in bashrc/profile
 * with "export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH$:/usr/include/eigen3"
 * Enter the Eigeh path to the includePath of VSCode to resolve the not found error
*/

// Matrix parameter is <Scalar, Rows, Cols>, where Scalar is the type of the matrix, Rows is the number of rows, and Cols is the number of columns
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
#if DEBUG_PRINT
    cout << "x is now:\t";
#endif
    while (x > 0)
    {
#if DEBUG_PRINT
        cout << x << "\t";
#endif

        if (x % 2 == 1)
            result *= A;
        A *= A;
        x /= 2; // Same as x >>= 1
    }
#if DEBUG_PRINT
    cout << endl;
#endif
    return result;
}

bool vector_approx_equal(Vec a, Vec b, double small_value)
{
    if (a.rows() != b.rows())
        return false;
    return ((a - b).norm() < small_value); // L2-Norm of vector
}

Vec pageRank_power_iter_naive(Mat A, Vec v_original)
{
    Vec result(v_original.rows());
    result.setZero();
    Vec prevVector = v_original;
    for (int i = 0; i < MAX_ITER; i++)
    {
        A = matrix_power_smart(A, i + 1);
        result = A * v_original;
        if (vector_approx_equal(result, prevVector, EPSILON))
        {
            cout << "Converged @ Iteration " << i << ": " << result.transpose() << endl;
            return result;
        }
#if DEBUG_PRINT
        cout << "Iter " << i << "'s vector v:\n"
             << result.transpose() << endl;
#endif
        prevVector = result;
    }
    // while (!vector_approx_equal(result, v, 0.0001))
    // {
    //     result = A * v;
    // }
    cout << "Converged @ Iteration " << MAX_ITER << " (Failed to Converge): " << result.transpose() << endl;
    throw exception();
    return result;
}

Vec pageRank_power_iter_naive_start(Mat A, int vector_length)
{
    // Initialize V to a guess, since this method is not starting condition-sensitive
    Vec v(vector_length);
    // Since this is PageRank, the vectors have to add up to 1.0
    long double value = (long double)1 / vector_length;
    // value = 1;
    for (int i = 0; i < vector_length; i++)
    {
        v(i) = value;
    }
    cout << "Starting vector: " << v.transpose() << endl;
    return pageRank_power_iter_naive(A, v);
}

Vec pageRank_power_iter_modified(Mat A, Vec v_original, Vec tp)
{
    Vec result(v_original.rows());
    result.setZero();
    Vec prevVector = v_original;
    for (int i = 0; i < MAX_ITER; i++)
    {
        // A = matrix_power_smart(A, i + 1);
        result = A * prevVector + tp;
        if (vector_approx_equal(result, prevVector, EPSILON))
        {
            cout << "Converged @ Iteration " << i << ": " << result.transpose() << endl;
            return result;
        }
#if DEBUG_PRINT
        cout << "Iter " << i << "'s vector v:\n"
             << result.transpose() << endl;
#endif
        prevVector = result;
    }
    cout << "Converged @ Iteration " << MAX_ITER << " (Failed to Converge): " << result.transpose() << endl;
    throw exception();
    return result;
}

Vec pageRank_power_iter_modified_start(Mat A, int vector_length, double alpha)
{
    // Initialize V to a guess, since this method is not starting condition-sensitive
    Vec v(vector_length);
    // Since this is PageRank, the vectors have to add up to 1.0
    long double value = (long double)1 / vector_length;
    // value = 1;
    for (int i = 0; i < vector_length; i++)
    {
        v(i) = value;
    }
    cout << "Starting vector: " << v.transpose() << endl;

    // Multiply the Matrix A by weight alpha
    A *= alpha;

    // Teleporting cofficients vector:
    Vec tp(vector_length);
    double beta = (1 - alpha) / vector_length;
    for (int i = 0; i < vector_length; i++)
    {
        tp(i) = beta;
    }

    return pageRank_power_iter_modified(A, v, tp);
}

/** 
 * @brief PageRank modified solved using Power Iteration Serial
 * 
 * Referencing the Pseudocode in tutorial:
 * @cite Mike Koltsov 
 * @link https://www.hackerearth.com/practice/notes/matrix-exponentiation-1/ @endlink
 * 
 * Run with "g++ $(pkg-config --cflags eigen3) pageRank_power_iter_serial.cpp -o pageRank_power_iter_serial && ./pageRank_power_iter_serial"
 * OR use the Makefile
 * @link https://stackoverflow.com/questions/21984971/how-to-compile-a-c-program-using-eigen-without-specifying-the-i-flag @endlink
 */
int main()
{
    try
    {
        cout << "Serial Version of PageRank Power Iteration/Method Linear Solver" << endl;
        // Mat A(2, 2);
        // A << 1, 2, 3, 4;
        // cout << "Matrix A sizes is rows: " << A.rows() << " and columns: " << A.cols() << endl;
        // Mat B = matrix_power_naive(A, 2);
        // cout << "Matrix B: " << endl
        //      << B << endl;
        // Mat C = matrix_power_smart(A, 7);
        // cout << "Matrix C: " << endl
        //      << C << endl;

        // Mat M(3, 3);
        // M << 0.5, 0.5, 0, 0.5, 0, 1, 0, 0.5, 0;
        // cout << "Matrix M: " << endl
        //      << M << endl;
        // Vec rst = pageRank_power_iter_naive_start(M, M.rows());
        // cout << "Vector result of M * v_o is:\n"
        //      << rst << endl;

        Mat M2(3, 3);
        M2 << 0.5, 0.5, 0, 0.5, 0, 0, 0, 0.5, 1;
        cout << "Matrix M2: " << endl
             << M2 << endl;
        Vec rst2 = pageRank_power_iter_naive_start(M2, M2.rows());
        cout << "Vector result of M * v_o is:\n"
             << rst2 << endl;

        // Mat M3(3, 3);
        // M3 << 0.5, 0.5, 0, 0.5, 0, 1, 0, 0.5, 0;
        // M3 *= 0.5;
        // cout << "Matrix M3 multiplied by 0.5: " << endl
        //      << M3 << endl;

        Vec rst3 = pageRank_power_iter_modified_start(M2, M2.rows(), TELEPORT_PARAMETER);
        cout << "Vector result of M * v_o using Modified PageRank is:\n"
             << rst3 << endl;
    }
    catch (exception const &e) // Solving warning: catching polymorphic type ‘class std::exception’ by value with "const &"
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