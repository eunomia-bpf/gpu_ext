CC ?= cc
CFLAGS ?= -O2 -g
CFLAGS += -std=gnu11 -Wall -Wextra -Werror

.PHONY: all test clean

all: tsc_ptimer_sanity

tsc_ptimer_sanity: tsc_ptimer_sanity.c
	$(CC) $(CFLAGS) $< -o $@

test: tsc_ptimer_sanity
	./tsc_ptimer_sanity --self-test

clean:
	$(RM) tsc_ptimer_sanity

