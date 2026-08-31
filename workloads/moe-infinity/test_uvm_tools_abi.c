/* Compile-only check of the monitor's wire layout against NVIDIA headers. */
#include <stddef.h>
#include "uvm_ioctl.h"

_Static_assert(UVM_TOOLS_INIT_EVENT_TRACKER_V2 == 76, "init ioctl");
_Static_assert(UVM_TOOLS_SET_NOTIFICATION_THRESHOLD == 57, "threshold ioctl");
_Static_assert(UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS == 58, "enable ioctl");
_Static_assert(UvmEventTypeEviction == 14, "eviction event");
_Static_assert(UvmEventNumTypesAll == 64, "event count");
_Static_assert(sizeof(UvmEventEntry_V2) == 72, "event stride");
_Static_assert(sizeof(UvmToolsEventControlData) == 528, "control size");
_Static_assert(offsetof(UvmToolsEventControlData, dropped) == 16, "drop counters");
_Static_assert(sizeof(UVM_TOOLS_INIT_EVENT_TRACKER_V2_PARAMS) == 56, "init size");
_Static_assert(offsetof(UVM_TOOLS_INIT_EVENT_TRACKER_V2_PARAMS, processor) == 24, "uuid");
_Static_assert(offsetof(UVM_TOOLS_INIT_EVENT_TRACKER_V2_PARAMS, allProcessors) == 40, "all processors");
_Static_assert(offsetof(UVM_TOOLS_INIT_EVENT_TRACKER_V2_PARAMS, uvmFd) == 44, "uvm fd");
_Static_assert(offsetof(UVM_TOOLS_INIT_EVENT_TRACKER_V2_PARAMS, rmStatus) == 48, "init status");
_Static_assert(sizeof(UVM_TOOLS_SET_NOTIFICATION_THRESHOLD_PARAMS) == 8, "threshold size");
_Static_assert(sizeof(UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS_PARAMS) == 16, "enable size");
_Static_assert(offsetof(UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS_PARAMS, rmStatus) == 8, "enable status");
_Static_assert(sizeof(UvmEventEvictionInfo_V2) == 40, "eviction size");
_Static_assert(offsetof(UvmEventEvictionInfo_V2, srcIndex) == 4, "source processor");
_Static_assert(offsetof(UvmEventEvictionInfo_V2, dstIndex) == 6, "destination processor");
_Static_assert(offsetof(UvmEventEvictionInfo_V2, addressOut) == 8, "address out");
_Static_assert(offsetof(UvmEventEvictionInfo_V2, addressIn) == 16, "address in");
_Static_assert(offsetof(UvmEventEvictionInfo_V2, size) == 24, "evicted bytes");
_Static_assert(offsetof(UvmEventEvictionInfo_V2, timeStamp) == 32, "timestamp");
