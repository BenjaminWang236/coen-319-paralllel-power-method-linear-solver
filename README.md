# coen-319-paralllel-power-method-linear-solver

COEN 319 Parallel Computing (SCU F2021)

Parallelization of the PageRank algorithm solved with Power Iteration algorithm in C++ 

Using OpenMP (OMP) and C++ Native Threads (NT)

Environment:
    1) GCCcore/10.2.0   2) zlib/1.2.11-GCCcore-10.2.0   3) binutils/2.35-GCCcore-10.2.0   4) GCC/10.2.0   5) Eigen/3.3.7
HPC:    WAVE
Compilation: make clean && make

Installation: None required

Run: <program> <graph_file> <pagerank_file> <-t> <num_threads>
OMP:    ./pageRank_power_iter_omp ./test/chvatal.txt ./test/demo1-pr.txt -t 1
NT:     ./pageRank_power_iter_nt ./test/chvatal.txt ./test/demo1-pr.txt -t 1

Note that the -t and num_threads are not optional parameters. Also the pagerank_file could be any string as it is not used
