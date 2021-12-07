OMPSRC := pageRank_power_iter_openMP.cpp
NTSRC := pageRank_power_iter_nativeThreads.cpp
# INC=-I/WAVE/users/unix/bwang4/eigen/Eigen
INC=$(pkg-config --cflags eigen3)

default: pageRank_power_iter_omp pageRank_power_iter_nt

pageRank_power_iter_omp: $(OMPSRC)
	g++ -fopenmp -O3 -Wall -Wextra $(INC)  -o $@ $<

pageRank_power_iter_nt: $(NTSRC)
	g++ -O3 -Wall -Wextra -Wno-cast-function-type $(INC) -pthread -fopenmp -o $@ $<

clean: 
	rm -vf pageRank_power_iter_omp pageRank_power_iter_nt
