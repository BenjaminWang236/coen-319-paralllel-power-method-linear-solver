#include <iostream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <cmath>
#include <omp.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>

#include <errno.h>
#include <dirent.h>

#define MAX_ITER 50
#define EPSILON 0.0001
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

using namespace std;
using namespace Eigen;

long double l2_norm(Vec v)
{
    // Parallel L2 Norm:
    // L2 norm is the square root of the sum of the squares of the elements of a vector
    // Parallelizing the summation of the squares of the elements of a vector and the squaring of the elements of a vector
    // Not parallelizing the square-root operation
    long double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < v.rows(); i++)
    {
        sum += v(i) * v(i);
    }

    // return (a - b).norm();
    return sqrt(sum);
}

bool vector_approx_equal(Vec a, Vec b, double small_value)
{
    if (a.rows() != b.rows())
    {
        cout << "ERROR: vector_approx_equal: a.rows()" << a.rows() << "!= b.rows()" << b.rows() << endl;
        throw exception();
        return false;
    }
    return (l2_norm(a - b) < small_value); // L2-Norm of vector
}

Vec matrix_vector_multiply(Mat A, Vec x)
{
    if (A.cols() != x.rows())
    {
        cout << "ERROR: matrix_vector_multiply: A.cols()" << A.cols() << "!= x.rows()" << x.rows() << endl;
        throw exception();
        return Vec();
    }

    // Parallel Matrix-Vector Multiplication:
    Vec result = Vec::Zero(A.rows());
    // Calculate the tile size: each thread will calculate a tile of the result
    // Tile size = number of rows of A/result divided by the number of threads
    // Last tile might have less than full tile-size
    for (int i = 0; i < result.rows(); i++)
    {
        // result(i) = A.row(i).dot(x);
        long double sum = 0.0;
        for (int j = 0; j < x.rows(); j++)
        {
            sum += A(i, j) * x(j);
        }
        result(i) = sum;
    }

    // result = A * x;
    // return A * x;
    return result;
}

Vec pageRank_power_iter_modified(Mat A, Vec v_original, Vec tp)
{
    Vec result = Vec::Zero(v_original.rows());
    Vec prevVector = v_original;
    for (int i = 0; i < MAX_ITER; i++)
    {
        // A = matrix_power_smart(A, i + 1);
        // result = A * prevVector + tp;
        result = matrix_vector_multiply(A, prevVector) + tp;
        if (vector_approx_equal(result, prevVector, EPSILON))
        {
#if DEBUG_PRINT
            cout << "Converged @ Iteration " << i << ": " << result.transpose() << endl;
#else
            cout << "Converged @ Iteration " << i << endl;
#endif
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
    double beta = (1 - alpha) / vector_length;
    // Vec tp = Vec::Constant(vector_length, beta);
    auto t1 = omp_get_wtime();
    Vec rst = pageRank_power_iter_modified(A, v, Vec::Constant(vector_length, beta));
    auto t2 = omp_get_wtime();
    cout << "Execution Time: " << (t2 - t1) << endl;
    return rst;
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
int main(int argc, char *argv[])
{
    try
    {
        cout << "OpenMP Version of PageRank Power Iteration/Method Linear Solver" << endl;
#if DEBUG_PRINT
        cout << "argc:\t" << argc << endl;
        for (int i = 0; i < argc; i++)
        {
            cout << "argv[" << i << "]:\t" << argv[i] << endl;
        }
#endif


        if (argc != 5)
        {
            cerr << "Invalid options." << endl
             << "<program> <graph_file> <pagerank_file> <-t> <num_threads>" << endl;
            exit(1);
        }
        string graph_file = argv[1];
        string pagerank_file = argv[2];
        int nthreads = atoi(argv[4]);
        omp_set_num_threads(nthreads);  // auto t1 = omp_get_wtime();

        // Read the graph file:
        // Read into a Matrix of Row-From Column-To as Binary 1/0
        // Transpose the Matrix to get the Column-From Row-To, since we're counting in-Links
        // To convert into the linear system of equations,
        // Divide each each element of a ROW of transposed Matrix by the sum of the COLUMN
        // Each ROW corresponds to an equation of a graph node in the pageRank equation

        Mat M(3, 3);
        M << 0.5, 0.5, 0, 0.5, 0, 0, 0, 0.5, 1;
        cout << "Matrix M:" << endl
             << M << endl;
        Vec rst = pageRank_power_iter_modified_start(M, M.rows(), TELEPORT_PARAMETER);
        cout << "Vector result of M * v_o using Modified PageRank is:\n"
             << rst << endl;
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