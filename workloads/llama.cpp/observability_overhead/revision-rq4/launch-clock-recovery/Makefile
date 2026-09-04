CC ?= cc
CFLAGS ?= -O2 -g
CPPFLAGS ?=

CFLAGS += -std=gnu11 -Wall -Wextra -Werror

.PHONY: all clean

all: rm_ptimer_correlation_sanity

rm_ptimer_correlation_sanity: rm_ptimer_correlation_sanity.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@

clean:
	$(RM) rm_ptimer_correlation_sanity
