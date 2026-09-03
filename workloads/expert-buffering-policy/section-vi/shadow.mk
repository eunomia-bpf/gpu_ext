# Independent, CPU-only correctness bridge; never rebuild the real selector.
CXX := /usr/bin/g++-13
CPUSET ?= 17
.PHONY: shadow test-shadow
shadow: build/libeb_shadow.so

build/libeb_shadow.so: shadow_bridge.cpp policy.h
	mkdir -p build
	taskset -c $(CPUSET) $(CXX) -std=c++17 -O2 -fPIC -shared -Wall -Wextra -Werror -Wl,--build-id=none $< -ldl -pthread -o $@

test-shadow: shadow
	taskset -c $(CPUSET) /usr/bin/python3 -B test_shadow_bridge.py
