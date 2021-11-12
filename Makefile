CXX=g++

CXXFLAGS=-std=c++14 \
  -Ofast -march=native \
  -Wall -Wextra -Wpedantic -Werror -fopenmp \

esch: esch.o
	$(CXX) $(CXXFLAGS) esch.o -o esch

esch.o: esch.cpp
	$(CXX) $(CXXFLAGS) -c esch.cpp

clean:
	rm esch esch.o
