// hb_map_abi.h: shared ABI for the Hummingbird original host+device split.
//
// The host tiles a CUDA launch into device subgrids (a "tile"). Each thread
// block of the split cubin computes the original-grid block index by adding
// the host-provided tile offset to its local split-block index:
//
//     out = block(local blockIdx) + off(hb_offset_*)
//
// The exact algorithm is shared by two device engines so the mapped
// coordinates that drive the resnet152 kernels are bit-identical:
//   * HB_ENGINE_NATIVE: inline native PTX (the existing device path).
//   * HB_ENGINE_BPF: a real eBPF program compiled to PTX by the GPU compiler.
//
// The context is a fixed-size scalar snapshot materialized in kernel local
// memory by the trusted wrapper. The BPF program dereferences no nested
// device pointer; it reads the in-* fields and writes the out-*/status/engine
// fields at fixed offsets, all below 48 bytes from the context pointer. The
// exporter is built to verify against that 48-byte PREVAIL context.
//
// This header is compiled both by clang -target bpf and by nvcc, so it must
// stay free of C++-only and CUDA-only constructs.

#ifndef HB_MAP_ABI_H
#define HB_MAP_ABI_H

typedef unsigned int hb_u32;
typedef unsigned long long hb_u64;

#define HB_MAP_ABI_VERSION 1u

// Which engine produced the mapped coordinates in out_*.
#define HB_ENGINE_NATIVE 1u
#define HB_ENGINE_BPF 2u

// status returned in the context by the device engine.
#define HB_MAP_OK 1u
#define HB_MAP_BAD 2u

// 12 x u32 = 48 bytes. The ctx48 exporter verifies against this 48-byte bound.
// Every field the device engine touches lives below 48 bytes from the context
// pointer.
struct HbMapContext {
    hb_u32 block_x, block_y, block_z; // in: local split-block index (blockIdx)
    hb_u32 off_x, off_y, off_z;       // in: host tile offset (hb_offset_*)
    hb_u32 out_x, out_y, out_z;       // out: original-grid block index
    hb_u32 status;                    // out: HB_MAP_OK
    hb_u32 engine;                    // out: HB_ENGINE_NATIVE | HB_ENGINE_BPF
    hb_u32 reserved[1];
};

#ifdef __cplusplus
static_assert(sizeof(struct HbMapContext) == 48,
              "HbMapContext must be exactly 48 bytes");
#else
_Static_assert(sizeof(struct HbMapContext) == 48,
               "HbMapContext must be exactly 48 bytes");
#endif

#endif // HB_MAP_ABI_H
