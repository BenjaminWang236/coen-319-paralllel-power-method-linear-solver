SRC := pageRank_power_iter_serial.cpp
OMPSRC := pageRank_power_iter_openMP.cpp
NTSRC := pageRank_power_iter_nativeThreads.cpp

default: pageRank_power_iter

pageRank_power_iter: $(SRC)
	g++ -g -O3 -Wall -Wextra $(pkg-config --cflags eigen3) -o $@ $<

pageRank_power_iter_omp: $(OMPSRC)
	g++ -fopenmp -O3 -Wall -Wextra -o $@ $<

pageRank_power_iter_nt: $(NTSRC)
	# mpic++ -O3 -Wall -Wextra -Wno-cast-function-type -o $@ $<

clean: 
	rm -vf pageRank_power_iter pageRank_power_iter_omp pageRank_power_iter_nt
