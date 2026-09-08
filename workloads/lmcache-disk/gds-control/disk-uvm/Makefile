NVCC    ?= nvcc
CFLAGS  ?= -std=c++14 -O2
TARGET  ?= disk_uvm_perf

$(TARGET): disk_uvm_perf.cu
	$(NVCC) $(CFLAGS) -o $@ $< -lcudart

clean:
	rm -f $(TARGET)

.PHONY: clean
