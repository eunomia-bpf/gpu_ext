CC ?= cc
CFLAGS ?= -O2 -g -Wall -Wextra -Werror

.PHONY: all clean

all: expert_eviction_monitor

expert_eviction_monitor: expert_eviction_monitor.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	$(RM) expert_eviction_monitor
