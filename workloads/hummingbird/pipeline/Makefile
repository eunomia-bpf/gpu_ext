HB_CPUSET ?= 17
PRIVATE := $(CURDIR)/build
.PHONY: prepare core test-cpu test-runner
prepare:
	taskset -c $(HB_CPUSET) python3 -B prepare.py

# Host compilation/link only: reuse the frozen split cubin and existing libraries.
core: prepare
	$(MAKE) -C $(PRIVATE)/src -j1 HB_CPUSET=$(HB_CPUSET) $(PRIVATE)/hummingbird_client $(PRIVATE)/idle_policy.bin

test-cpu: prepare
	$(MAKE) -C $(PRIVATE)/src -j1 HB_CPUSET=$(HB_CPUSET) $(PRIVATE)/test_idle_policy $(PRIVATE)/test_profile $(PRIVATE)/idle_policy.bin
	taskset -c $(HB_CPUSET) $(PRIVATE)/test_idle_policy $(PRIVATE)/idle_policy.bin
	taskset -c $(HB_CPUSET) $(PRIVATE)/test_profile $(PRIVATE)/idle_policy.bin
	taskset -c $(HB_CPUSET) /usr/bin/g++-13 -std=c++17 -O2 -Wall -Wextra -Werror -Wl,--build-id=none test_pipeline_events.cpp -o $(PRIVATE)/test_pipeline_events
	taskset -c $(HB_CPUSET) $(PRIVATE)/test_pipeline_events

test-runner: prepare
	taskset -c $(HB_CPUSET) python3 -B test_pipeline_study.py -v
