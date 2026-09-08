# Invocation-count diagnostic: initial attempt

Runtime `7cfba18` builds successfully. Count instrumentation is opt-in and
separate from the completed 180 performance cells. The first diagnostic uses
two CTAs, 128 threads/CTA, two launches, zero warmup, automatic execution off.
Application and loader both complete with exit zero.

The application logs an 8-byte device counter global, but no observed count
is read back. The module destructor is not reached on this process-exit path;
the initial implementation only reads the counter there. `diagnostic.log`
therefore records an empty observation list and stops, rather than presenting
launch geometry as measured calls. No performance cell was repeated.

The local-Qwen follow-up moves opt-in readback to a live-context patched-launch
completion point. These original records remain; the next diagnostic will use
a fresh directory. Timing figures from counter-instrumented runs are not
performance results.
