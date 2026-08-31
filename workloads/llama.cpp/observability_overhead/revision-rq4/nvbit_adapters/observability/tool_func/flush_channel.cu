#include "utils/channel.hpp"

extern "C" __global__ void flush_channel(ChannelDev* channel) {
    channel->flush();
}
