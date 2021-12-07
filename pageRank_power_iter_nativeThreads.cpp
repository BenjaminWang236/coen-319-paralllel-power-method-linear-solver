#include <iostream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <thread>
#include <vector>

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

using namespace std;
using namespace Eigen;


void mult(Vec* result, Mat* A, Vec* prevVector, int rstart, int rend, int cstart, int cend, int qstart, int qend) {
    int r,c,q;
    for(r = rstart; r <= rend; r++) {
        for(c = cstart; c <= cend; c++) {
            if(qstart == 0) {
                result->coeffRef(r,c) = 0.0;
            }
            for(q = qstart; q <= qend; q++) {
                result->coeffRef(r,c) = result->coeffRef(r,c) + A->coeffRef(r,q) * prevVector->coeffRef(q,c);
            }
        }
    }
}

bool vector_approx_equal(Vec a, Vec b, double small_value)
{
    if (a.rows() != b.rows())
        return false;
    return ((a - b).norm() < small_value); // L2-Norm of vector
}

Vec pageRank_power_iter_modified(Mat A, Vec v_original, Vec tp)
{
    Vec result(v_original.rows());
    // Vec ojas_test(v_original.rows());
    int r_tile_size = 2;
    int c_tile_size = 2;
    int q_tile_size = 2;
    vector<thread> th_vec;
    result.setZero();
    Vec prevVector = v_original;
    for (int i = 0; i < MAX_ITER; i++)
    {
        // A = matrix_power_smart(A, i + 1);
        // result = A * prevVector + tp;
        for(int rstart = 0; rstart < A.rows(); rstart += r_tile_size) {
            int rend = rstart + r_tile_size - 1;
            if(rend >= A.rows()) {
                rend = A.rows() - 1;
            }
            for(int cstart = 0; cstart < prevVector.cols(); cstart += c_tile_size) {
                int cend = cstart + c_tile_size - 1;
                if(cend >= prevVector.cols()) {
                    cend = prevVector.cols() - 1;
                }
                for(int qstart = 0; qstart < A.cols(); qstart += q_tile_size) {
                    int qend = qstart + q_tile_size - 1;
                    if(qend >= A.cols()) {
                        qend = A.cols() - 1;
                    }
                    thread th (mult, &result, &A, &prevVector, rstart, rend, cstart, cend, qstart, qend);
                    th_vec.push_back(move(th));
                    if(th_vec.size() == 10) {
                        for(int i = 0; i < th_vec.size(); i++) {
                            th_vec[i].join();
                        }
                        th_vec.clear();
                    }
                    if(qend = A.cols() - 1) {
                        for(int i = 0; i < th_vec.size(); i++) {
                            th_vec[i].join();
                        }
                        th_vec.clear();
                    }
                }
            } 
        } // end of final for
        for(int i = 0; i < th_vec.size(); i++) {
            th_vec[i].join();
        }
        th_vec.clear();
        // ojas_test = ojas_test + tp;
        result = result + tp;
        // cout << ojas_test << endl;
        // cout << "result \n" << endl;
        // cout << result << endl;
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
        cout << "C++ Native Threads Version of PageRank Power Iteration/Method Linear Solver" << endl;

        Mat M(3, 3);
        M << 0.5, 0.5, 0, 0.5, 0, 0, 0, 0.5, 1;
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