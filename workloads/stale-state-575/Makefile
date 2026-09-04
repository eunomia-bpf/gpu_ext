CC ?= cc
NVCC ?= nvcc
CFLAGS ?= -O2 -g -Wall -Wextra -Werror
NVCCFLAGS ?= -O3 -std=c++17 -Xlinker --build-id=none

.PHONY: test-offline build-sources clean

test_policy_model: test_policy_model.c stale_state_policy_model.h
	$(CC) $(CFLAGS) -std=c11 test_policy_model.c -o $@

uvm_event_monitor: uvm_event_monitor.c
	$(CC) $(CFLAGS) -std=c11 uvm_event_monitor.c -o $@

stale_state_workload: stale_state_workload.cu
	$(NVCC) $(NVCCFLAGS) stale_state_workload.cu -o $@

test-offline: test_policy_model uvm_event_monitor
	./test_policy_model
	PYTHONDONTWRITEBYTECODE=1 python3 -B test_offline.py

build-sources: stale_state_workload uvm_event_monitor test_policy_model

clean:
	rm -f stale_state_workload uvm_event_monitor test_policy_model
