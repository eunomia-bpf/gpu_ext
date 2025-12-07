# Simple Makefile for UVM Microbenchmark

.PHONY: all clean test

all: uvmbench

uvmbench: main.cu kernels/synthetic.cuh
	nvcc -arch=sm_120 -O3 -I. -o uvmbench main.cu -lcudart

clean:
	rm -f uvmbench *.o

test: uvmbench
	./uvmbench --kernel=seq_stream --mode=device --size_factor=0.05 --iterations=3
	./uvmbench --kernel=rand_stream --mode=uvm --size_factor=0.05 --iterations=3
	./uvmbench --kernel=pointer_chase --mode=uvm_prefetch --size_factor=0.05 --iterations=3
