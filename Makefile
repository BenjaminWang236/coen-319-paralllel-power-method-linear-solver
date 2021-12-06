SRC := pageRank_power_iter_serial.cpp
OMPSRC := pageRank_power_iter_openMP.cpp
NTSRC := pageRank_power_iter_nativeThreads.cpp

# INC=-I/WAVE/users/unix/bwang4/eigen/Eigen
INC=$(pkg-config --cflags eigen3)

# default: pageRank_power_iter pageRank_power_iter_omp
default: pageRank_power_iter_omp

pageRank_power_iter: $(SRC)
	g++ -g -O3 -Wall -Wextra $(INC) -o $@ $<

pageRank_power_iter_omp: $(OMPSRC)
	g++ -g -fopenmp -O3 -Wall -Wextra $(INC)  -o $@ $<

pageRank_power_iter_nt: $(NTSRC)
	# mpic++ -O3 -Wall -Wextra -Wno-cast-function-type $(INC) -o $@ $<

clean: 
	# rm -vf pageRank_power_iter pageRank_power_iter_omp pageRank_power_iter_nt
	rm -vf pageRank_power_iter_omp
