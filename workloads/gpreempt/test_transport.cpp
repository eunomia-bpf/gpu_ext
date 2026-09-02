// CPU-only ABI tests. Linker wrappers prevent opening or issuing GPU ioctls.
#include "gpreempt.h"
#include <cassert>
#include <cerrno>
#include <cstdarg>
#include <cstdio>

extern thread_local int fd;
enum class Reply { ok, syscall_error, status_error, no_object, no_channels, too_many };
static Reply reply = Reply::ok;
static unsigned calls = 0;
static unsigned long last_operation;
static unsigned last_flags;
static unsigned long long last_timeslice;

extern "C" int __wrap_open(const char *, int flags, ...)
{
    assert(flags & O_CLOEXEC);
    return 77;
}

extern "C" int __wrap_ioctl(int descriptor, unsigned long operation, ...)
{
    assert(descriptor == 77);
    va_list ap;
    va_start(ap, operation);
    auto *args = va_arg(ap, NVOS54_PARAMETERS *);
    va_end(ap);
    ++calls;
    last_operation = operation;
    last_flags = args->flags;
    assert(args->status == 0);
    if (reply == Reply::syscall_error) { errno = EINVAL; return -1; }
    if (reply == Reply::status_error) { args->status = 123; return 0; }
    if (operation == OP_QUERY && args->flags == 0) {
        assert(args->hClient == 321 && args->hObject == 0 && args->cmd == 0);
        assert(args->paramsSize == sizeof(NvChannels));
        auto *channels = static_cast<NvChannels *>(args->params);
        assert(channels->numChannels == 0);
        args->hClient = 111;
        args->hObject = reply == Reply::no_object ? 0 : 222;
        channels->numChannels = reply == Reply::no_channels ? 0 :
                                reply == Reply::too_many ? 65 : 2;
    } else if (operation == OP_QUERY) {
        assert(args->flags == 0x00010001 && args->cmd == NVA06C_CTRL_CMD_SET_TIMESLICE);
        assert(args->hClient == 111 && args->hObject == 222 && args->paramsSize == 8);
        last_timeslice = static_cast<NVA06C_CTRL_TIMESLICE_PARAMS *>(args->params)->timesliceUs;
    } else {
        assert(operation == OP_CONTROL && args->flags == 0);
    }
    return 0;
}

int main()
{
    assert(NvRmQuery(nullptr) == NV_ERR_GENERIC);
    NvContext invalid{};
    assert(NvRmQuery(&invalid) == NV_ERR_GENERIC && calls == 0);
    for (Reply mode : {Reply::syscall_error, Reply::status_error, Reply::no_object,
                       Reply::no_channels, Reply::too_many}) {
        reply = mode;
        NvContext query{};
        query.hClient = 321;
        assert(NvRmQuery(&query) != NV_OK);
        assert(query.hClient == 0 && query.hObject == 0);
    }
    reply = Reply::ok;
    NvContext context{};
    context.hClient = 321;
    assert(NvRmQuery(&context) == NV_OK);
    assert(context.hClient == 111 && context.hObject == 222 && context.channels.numChannels == 2);
    assert(set_priority(context, 0) == 0 && last_timeslice == 1000000);
    assert(last_operation == OP_QUERY && last_flags == 0x00010001);
    assert(set_priority(context, 1) == 0 && last_timeslice == 1);
    reply = Reply::syscall_error;
    assert(set_priority(context, 0) == -1);
    reply = Reply::status_error;
    assert(set_priority(context, 1) == -1);
    reply = Reply::ok;
    unsigned long long parameter = 1;
    assert(NvRmControl(111, 222, 1234, &parameter, sizeof(parameter)) == NV_OK);
    assert(last_operation == OP_CONTROL && last_flags == 0);
    reply = Reply::syscall_error;
    assert(NvRmControl(111, 222, 1234, &parameter, sizeof(parameter)) == NV_ERR_GENERIC);
    reply = Reply::status_error;
    assert(NvRmControl(111, 222, 1234, &parameter, sizeof(parameter)) == 123);
    assert(NvRmModifyTS(invalid, 1) == NV_ERR_GENERIC);
    assert(NvRmModifyTS(context, 0) == NV_ERR_GENERIC);
    std::printf("{\"test\":\"gpreempt_575_transport\",\"gpu_execution\":false,\"ioctls_mocked\":%u,\"passed\":true}\n", calls);
}
