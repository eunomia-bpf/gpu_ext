#include <stdint.h>

#include "utils/channel.hpp"

extern "C" __global__ void flush_channel(ChannelDev* channel) {
    channel->flush();
}

extern "C" __global__ void sample_globaltimer(uint64_t* output) {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    *output = value;
}
