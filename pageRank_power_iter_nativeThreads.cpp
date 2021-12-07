#include <iostream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <thread>
#include <vector>
#include <fstream>
#include <sstream>
#include <omp.h>

#define MAX_ITER 50
#define EPSILON 0.0000001
#define DEBUG_PRINT 0 // 0: no print, 1: print
#define TELEPORT_PARAMETER 0.8
int nthreads = 1;

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

// function that each thread executes
// takes in a pointer vec result, pointer matrix A, and pointer vec prevVector
// these hold the result and the values needed for the mutliplication
// rstart, rend, cstart, cend, qstart, and qend are used as indices to result, A, and prevVector
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
    int r_tile_size = 2;
    int c_tile_size = 2;
    int q_tile_size = 2;
    vector<thread> th_vec;
    result.setZero();
    Vec prevVector = v_original;
    // code for tiled matrix vector multiplication
    for (int i = 0; i < MAX_ITER; i++)
    {
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
                    // create thread to do assigned tile multiplication
                    thread th (mult, &result, &A, &prevVector, rstart, rend, cstart, cend, qstart, qend);
                    th_vec.push_back(move(th));
                    // check if there are nthreads number of threads in the threads vector
                    if(th_vec.size() == nthreads) {
                        // if so go and join all threads
                        for(int i = 0; i < th_vec.size(); i++) {
                            th_vec[i].join();
                        }
                        // clear threads vector when done
                        th_vec.clear();
                    }
                    // check if at end of A
                    if(qend = A.cols() - 1) {
                        // if so go and join all threads
                        for(int i = 0; i < th_vec.size(); i++) {
                            th_vec[i].join();
                        }
                        th_vec.clear();
                    }
                }
            } 
        } // end of final for
        // go through threads vector and make sure all remaining threads are joined
        for(int i = 0; i < th_vec.size(); i++) {
            th_vec[i].join();
        }
        // clear threads vector
        th_vec.clear();
        result = result + tp;
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
    auto t1 = omp_get_wtime();
    Vec rst =  pageRank_power_iter_modified(A, v, tp);
    auto t2 = omp_get_wtime();
    cout << "Execution Time: " << (t2 - t1) << endl;
    return rst;
}


Mat read_Graph_return_Matrix(string file_name)
{
    Mat m = Mat::Zero(1, 1);
    ifstream file(file_name);
    if (!file.is_open())
    {
        cout << "ERROR: Could not open file: " << file_name << endl;
        throw exception();
        return m;
    }
    string line;
    string delim = " ";
    int from = 0, to = 0;
    while (getline(file, line))
    {
#if DEBUG_PRINT
        cout << line << endl;
#endif
        string elem;
        stringstream ss(line);
        ss >> elem;
        from = atoi(elem.c_str());
        ss >> elem;
        to = atoi(elem.c_str());
        if (m.rows() < from + 1)
        {
            int orig_nrows = m.rows();
            m.conservativeResize(from + 1, from + 1);
            for (int i = orig_nrows; i < m.rows(); i++)
            {
                m.row(i).setZero();
            }
#if DEBUG_PRINT
            cout << "Row expanded m:\n"
                 << m << endl;
#endif
        }
        if (m.cols() < to + 1)
        {
            int orig_ncols = m.cols();
            m.conservativeResize(to + 1, to + 1);
            for (int i = orig_ncols; i < m.cols(); i++)
            {
                m.col(i).setZero();
            }
#if DEBUG_PRINT
            cout << "Column expanded m:\n"
                 << m << endl;
#endif
        }
        m(from, to) = 1;
    }
    file.close();
    if (m.rows() < m.cols())
    {
        int orig_nrows = m.rows();
        m.conservativeResize(m.cols(), m.cols());
        for (int i = orig_nrows; i < m.cols(); i++)
        {
            m.row(i).setZero();
        }
    }
    if (m.cols() < m.rows())
    {
        int orig_ncols = m.cols();
        m.conservativeResize(m.rows(), m.rows());
        for (int i = orig_ncols; i < m.rows(); i++)
        {
            m.col(i).setZero();
        }
    }
#if DEBUG_PRINT
    cout << "Matrix read from file:\n"
         << m << endl;
#endif

    // Transpose the Matrix to get the Column-From Row-To, since we're counting in-Links
    // To convert into the linear system of equations,
    // Divide each each element of a ROW of transposed Matrix by the sum of the COLUMN
    // Each ROW corresponds to an equation of a graph node in the pageRank equation
    m.transposeInPlace();
#if DEBUG_PRINT
    cout << "Transposed Matrix:\n"
         << m << endl;
#endif
    vector<long double> col_sums; // Number of Out-Links from a node, since Column is now From/Out-Links
    for (int i = 0; i < m.cols(); i++)
    {
        col_sums.push_back(m.col(i).sum());
#if DEBUG_PRINT
        cout << "col_sums[" << i << "] = " << col_sums[i] << endl;
#endif
    }
    for (int i = 0; i < m.rows(); i++)
    {
        for (int j = 0; j < m.cols(); j++)
        {
            if (col_sums[j] == 0)
                m(i, j) = 0;
            else
                m(i, j) /= col_sums[j];
        }
    }
#if DEBUG_PRINT
    cout << "Transposed Matrix with pageRank algorithm:\n"
         << m << endl;
#endif

    return m;
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
        cout << "Native Threads Version of PageRank Power Iteration/Method Linear Solver" << endl;
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
        nthreads = atoi(argv[4]);

        // Read the graph file:
        // NOTE: Run in terminal with "clear && make clean && make && ./pageRank_power_iter_omp ./test/demo1.txt ./test/demo1-pr.txt -t 1 > debug.txt"
        Mat mat = read_Graph_return_Matrix(graph_file);

        // Mat M(3, 3);
        // M << 0.5, 0.5, 0, 0.5, 0, 0, 0, 0.5, 1;  // demo1.txt
        Vec rst = pageRank_power_iter_modified_start(mat, mat.rows(), TELEPORT_PARAMETER);
        cout << "Vector result of M * v_o using Modified PageRank is:\n"
             << rst << endl;
        cout << "rst sum: " << rst.sum() << endl;
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