// CPU-only test of the actual private-runtime header, not a CUDA attachment test.
#include "bootstrap_logger.hpp"
#include "bpftime_vm_compat.hpp"
#include <cstdio>
#include <memory>

static std::unique_ptr<bpftime::vm::compat::bpftime_vm_impl> factory()
{
    return nullptr;
}

int main()
{
    bpftime::initialize_agent_bootstrap_logger();
    bpftime::initialize_agent_bootstrap_logger(); // repeated fatbin registration
    bpftime::vm::compat::register_vm_factory("bootstrap-output-test", factory);
    bpftime::vm::compat::register_vm_factory("bootstrap-output-test", factory);
    std::puts("application-output");
    bpftime::bpftime_logger_flush();
}
