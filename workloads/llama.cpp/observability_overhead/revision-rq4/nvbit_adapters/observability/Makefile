NVCC := nvcc -ccbin=$(CXX) -D_FORCE_INLINES
PTXAS := ptxas
BIN2C := bin2c

NVBIT_ROOT ?= ../../../deps/nvbit_release_x86_64
NVBIT_PATH := $(NVBIT_ROOT)/core
INCLUDES := -I$(NVBIT_PATH)
LIBS := -L$(NVBIT_PATH) -lnvbit
NVCC_PATH := -L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))
NVCC_OPT ?= -O3
ARCH ?= sm_120

SOURCES := observability.cu inject_funcs.cu
OBJECTS := $(SOURCES:.cu=.o)
TOOL_FUNC_SOURCES := $(wildcard tool_func/*.cu)
TOOL_FUNC_FATBINS := $(TOOL_FUNC_SOURCES:.cu=.fatbin)
TOOL_FUNC_BIN2CS := $(TOOL_FUNC_FATBINS:.fatbin=.c)

all: observability.so

observability.so: $(OBJECTS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) $(NVCC_OPT) $(OBJECTS) $(LIBS) $(NVCC_PATH) \
		-lcuda -lcudart_static -shared -o $@

tool_func/%.fatbin: tool_func/%.cu
	$(NVCC) -arch=$(ARCH) $(INCLUDES) $(NVCC_OPT) -fatbin $< -o $@

tool_func/%.c: tool_func/%.fatbin
	$(BIN2C) -c --name $(basename $(notdir $<))_bin $< > $@

observability.o: observability.cu $(TOOL_FUNC_BIN2CS)
	$(NVCC) -dc -c -std=c++14 $(INCLUDES) -Xptxas -cloning=no \
		-Xcompiler -Wall -arch=$(ARCH) $(NVCC_OPT) -Xcompiler -fPIC $< -o $@

inject_funcs.o: inject_funcs.cu
	$(NVCC) $(INCLUDES) -Xptxas -astoolspatch --keep-device-functions \
		-arch=$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

clean:
	rm -f observability.so *.o tool_func/*.fatbin tool_func/*.c

.PHONY: all clean
