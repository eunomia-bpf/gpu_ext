CC ?= cc
NVCC ?= nvcc
CFLAGS ?= -O2 -g
CPPFLAGS ?=

CFLAGS += -std=gnu11 -Wall -Wextra -Werror

.PHONY: all clean test

all: rm_ptimer_correlation_sanity rm_globaltimer_identity

rm_ptimer_correlation_sanity: rm_ptimer_correlation_sanity.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@

rm_ptimer_575.o: ../nvbit_adapters/observability/rm_ptimer_575.c \
		../nvbit_adapters/observability/rm_ptimer_575.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -c $< -o $@

rm_globaltimer_identity: rm_globaltimer_identity.cu rm_ptimer_575.o \
		../nvbit_adapters/observability/rm_ptimer_575.h
	$(NVCC) -std=c++14 -arch=sm_120 -Xcompiler=-Wall,-Wextra \
		$< rm_ptimer_575.o -o $@

test: rm_ptimer_correlation_sanity rm_globaltimer_identity
	./rm_ptimer_correlation_sanity --self-test
	./rm_globaltimer_identity --self-test

clean:
	$(RM) rm_ptimer_correlation_sanity rm_globaltimer_identity rm_ptimer_575.o
