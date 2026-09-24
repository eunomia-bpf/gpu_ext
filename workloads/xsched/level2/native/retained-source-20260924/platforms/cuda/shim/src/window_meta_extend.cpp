// SPDX-License-Identifier: GPL-2.0
//
// sm_120 compiler-declared extended-parameter meta extend (Level-2 native
// actuator port, additive and unintegrated until tested by the build owner).
//
// Opt-in with the XG_NATIVE_META_EXTEND environment variable (presence
// based). When enabled, every CUDA module image that passes through the
// module-load entry points of the shim is copied and, inside its sm_120
// cubins, every kernel whose declared constant-bank parameter region starts
// at the sm_120 parameter base 0x380 with a smaller extent gets that extent
// raised to 0x1520, which covers the fused guardian/resume parameter window
// at c[0x0][0x1880..0x189c] (blob-relative 0x1500..0x151c, see
// CudaKernelCommand::AdoptWindowRelayExtra and the 28-byte XG_ARGS_BYTES
// block in level2/xsched_guardian_abi.h).
//
// Module images arrive either bare (raw fatbin container 0xba55ed50 or ELF
// cubin) or wrapped in the CUDA runtime fatbin wrapper (fatbinary_section.h
// __fatBinC_Wrapper_t: u32 magic 0x466243b1, u32 version 1, data -> the
// real container, filename_or_fatbins). Static cudart hands the wrapper to
// cuLibraryLoadData and the driver follows wrapper->data (root gdb
// specimen). For a wrapped image the extender owns one block holding a
// wrapper copy whose data field is retargeted at the patched nested
// container copy inside the same block, so the driver receives a wrapper
// identical to the original except for the owned data pointer.
//
// Two compiler-written nvinfo records describe that extent and both are
// raised together (cuobjdump -elf equivalence, nvcc 12.9 specimen):
//   EIATTR_CBANK_PARAM_SIZE (id 0x19, fmt 3): 16-bit whole bank size
//       (0x20 for the specimen kernel) lives at record offset +2.
//   EIATTR_PARAM_CBANK     (id 0x0a, fmt 4): payload
//       {u32 kind 0xa/0x9, u16 param_start, u16 param_size}; the param
//       size is the 16-bit field at payload offset +6.
//
// A second opt-in (XG_NATIVE_META_KPARAM) also converts the KPARAM_INFO
// ordinal table into the compiler's own large-parameter shape and grows the
// max declared ordinal end so it covers the blob extent. Causal evidence
// (root HX4CV3 driver-entry control on the same installed binary, original
// entry selected): shortening the relay CU_LAUNCH_PARAM BUFFER_SIZE from
// 0x1520 to the original compute size 0x20 at the real driver
// cuLaunchKernel makes all 200 queued kernels launch ret 0 and the BE
// complete and exit cleanly, while the full-size relay fails the first
// launch with 701 (kv7jHZ, NmlhJN). The declared ordinal extent (0x20,
// from KPARAM_INFO) is therefore the candidate the driver checks against
// the buffer.
//
// Compiler specimen (nvcc 12.9 sm_120, kernel with a 0x1520-byte by-value
// struct param, /tmp/kpspec comparison): for that large parameter nvcc
// emits EIATTR_KPARAM_INFO_V2 (id 0x45, fmt 4, payload 12 bytes:
// {u32 index, u16 ordinal, u16 offset, u16 plain_size, u16 attrs}; attrs
// 0x0500 for pointer params, 0 for scalars), while small kernels of the
// same compilation keep the legacy EIATTR_KPARAM_INFO (id 0x17) records
// {index, ordinal, offset, u16 attrs (0xf500 ptr / 0xf000 scal), u16
// size_field == size*4+1 (observed on 5 records, sizes 4 and 8)}; it sets
// CBANK_PARAM_SIZE == PARAM_CBANK.psize == the max ordinal end
// (0x152c = last param end in the specimen) and sizes .nv.constant0 of the
// kernel to param_start + max ordinal end exactly (0x18ac; the workload
// kernels follow the same rule: 0x3a0 = 0x380 + 0x20). The in-place patch
// therefore keeps record slots at their lengths: legacy 0x17 records are
// converted in place to the V2 shape (plain size, attrs bits 12..15
// cleared, pointer/sanner mapping preserved: 0xf500 -> 0x0500, 0xf000 ->
// 0x0000) and the max-end ordinal size is grown to 0x1520.
//
// Since the extender works on images before function handles exist, it
// keeps the patched kernels' ORIGINAL per-ordinal layout, keyed by the
// kernel's mangled name (the same string cuFuncGetName reports). The HAL
// constructor deep copy and window-relay adoption look this layout up via
// XgGetRelayOriginalParams so every host-side marshal keeps the original
// sizes; only the driver's launch-side blob validation sees the grown
// extent, and non-relay/plain launches never marshal grown sizes.

#include "xsched/cuda/shim/window_meta_extend.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
//
// HOST-MARSHALING CONSTRAINT AND HAL WIRING: the grown ordinals must not
// be observed by the per-ordinal host deep copy in CudaKernelCommand's
// constructor: after the module is loaded with grown records,
// cuXtraGetParamInfo reports the grown size for the last ordinal and the
// constructor's memcpy of that size from the app's own 8-byte argument
// pointer would read ~5 KiB out of bounds BEFORE AdoptWindowRelayExtra
// ever runs. The constructor and the relay adoption therefore marshal -
// via the shim-registered ORIGINAL per-ordinal layout - with the original
// sizes (see XgGetRelayOriginalParams below); the padded 0x1520 blob is
// only materialized for blob-form launches, and non-relay launches never
// marshal grown sizes. Without XG_NATIVE_META_KPARAM records are left
// byte-identical and the HAL falls back to loaded param info.
// The KPARAM family conversion changes only the records named above; the
// KPARAM record SLOTS (count and ordinal order) are preserved, so
// per-argument ordinals/offsets keep their compiled layout; the window
// args remain extra, undeclared-by-KPARAM bank bytes inside the enlarged
// extent. All other records (EXIT_INSTR_OFFSETS 0x1c, SW_WAR 0x36, the
// .nv.merc mirrors, module-wide .nv.info) are left byte-identical.
//
// Pass D (same XG_NATIVE_META_KPARAM opt-in as pass C, constant0 growth):
// the compiler rule (bigparam specimen 0x18a8 == 0x380 + 0x1528; the
// workload kernels follow it: 0x3a0 == 0x380 + 0x20) sizes .nv.constant0
// to param_start + max ordinal end, and pass C raises the max ordinal end
// to 0x1520. Growing only the records leaves the bank DATA image at 0x3a0
// while the declared extent says 0x18a0, so the 0x1520-byte launch blob
// reaches past the section image the driver matches the param upload
// against (current first-launch 701 even under the original-entry control
// with resolver/lookup/marshaling verified good). Pass D therefore grows
// the kernel's .nv.constant0 section to 0x380 + 0x1520 = 0x18a0 by zero
// insertion at the section end. Because byte insertion shifts every later
// file byte, the ELF is repacked in the owned copy: the section's sh_size
// grows by the exact delta while all offsets move by the 16-byte-rounded
// shift (sh_offset of every later section, e_phoff, e_shoff, PT_LOAD
// p_offset/file-range, and the follower gaps stay alignment-clean since
// the shift is a multiple of 16 and all following sections are aligned <=
// 16). Inside a fatbin container the owning entry's padded-payload field
// grows by the shift, the trailing container bytes (later entries) move,
// and the container fileSize field (u64 at header +8) grows by the total.
// The insertion bytes are zero: c[0x0] content beyond the original param
// region has no compiler-defined image bytes (the launch parameter upload
// plus the relay window fill [0x1500,0x151c) cover everything the kernel
// reads).
//
// Reserve accounting: the owned copy carries, per pass-C-growable kernel,
// the maximum rounded shift (0x1520 bytes) so the insertion memmove always
// has destination room; the pre-pass counter overcounts (includes kernels
// whose walk fails later), never undercounts.
//
// Idempotence: an already-extended image reports extents >= 0x1520 and is
// left byte-identical (no ordinal growth, no constant0 growth), so any
// re-resolved load of the same already-patched image is unchanged.
//
// This is a CPU-side byte patch of the module image before the real driver
// loads it (cuModuleLoadData copies the image, so the mutated buffer can be
// freed right after the call). All failures pass the original image through
// unchanged: the extender never aborts the loading process.
//
// Ownership: the patched copy is a raw malloc block with a 16-byte header
// [uint64 magic 0x474d495458455847 ("XGEXTIMG") | uint64 data size] placed
// in front of the image bytes. The handed-out pointer is the image data
// pointer itself; XgFreeExtendedImage steps back over the header, validates
// the magic, and frees the whole block. For a wrapped image the block also
// holds the 24-byte wrapper copy in a 32-byte slot in front of the
// container copy; freeing the handed-out wrapper pointer frees the whole
// block. Pointers that do not carry the header are reported and
// deliberately NOT freed (leak beats crash); the header is read before
// validation, so XgFreeExtendedImage is NOT safe for arbitrary foreign
// pointers.

#include "xsched/utils/log.h"

namespace xsched::cuda {

namespace {

constexpr uint16_t kMetaParamStart = 0x380;  // sm_120 parameter-region base
constexpr uint16_t kMetaParamSize  = 0x1520; // covers window args rel 0x1500 + 0x1c, 16-aligned

// Compiler-shape .nv.constant0 extent for a grown kernel: param_start +
// grown max ordinal end (bigparam specimen 0x380 + 0x1528 = 0x18a8; the
// grown fused-window extent gives 0x380 + 0x1520 = 0x18a0).
constexpr size_t kMetaConst0Target = (size_t)kMetaParamStart + kMetaParamSize;
// Owned-copy reserve per pass-C-growable kernel: worst-case rounded shift
// is align16(0x18a0 - (0x380 + 4)) = 0x1520.
constexpr size_t kConst0GrowReserve = 0x1520;

// Little-endian stores mirroring the reads done by le(); all ELF/fatbin
// fields this patcher rewrites.
void put64(char *p, uint64_t v)
{
    for (int i = 0; i < 8; ++i) p[i] = (char)((v >> (8 * i)) & 0xff);
}

size_t align_up16(size_t v)
{
    return (v + 0xf) & ~(size_t)0xf;
}

// ---------------------------------------------------------------------
// Relay original-layout registry: for every kernel whose KPARAM table was
// grown (pass C), remember the app's ORIGINAL per-ordinal {ordinal,
// offset, size}, keyed by the kernel's mangled name (the same string
// cuFuncGetName reports). The HAL constructor deep copy and the window
// relay adoption consult this table through XgGetRelayOriginalParams, so
// host-side marshaling always uses the original sizes and only the
// driver's launch-side blob validation sees the grown extent.
constexpr size_t kRelayLayoutMaxKernels = 64;
constexpr size_t kRelayLayoutMaxParams  = 32;

struct RelayParamEntries
{
    size_t count;
    std::array<uint32_t, kRelayLayoutMaxParams> ordinals;
    std::array<uint32_t, kRelayLayoutMaxParams> offsets;
    std::array<uint32_t, kRelayLayoutMaxParams> sizes;
};

std::mutex g_relay_layout_mutex;
std::unordered_map<std::string, RelayParamEntries> g_relay_layouts;
bool g_relay_layout_overflow = false;

void remember_relay_originals(const char *kernel_name,
                              const std::vector<std::array<uint32_t, 3>> &entries)
{
    std::lock_guard<std::mutex> lock(g_relay_layout_mutex);
    if (g_relay_layouts.size() >= kRelayLayoutMaxKernels) {
        if (!g_relay_layout_overflow) {
            g_relay_layout_overflow = true;
            XWARN("meta-extend: relay layout registry full (%zu kernels); "
                  "further grown kernels marshal from loaded param info",
                  kRelayLayoutMaxKernels);
        }
        return;
    }
    if (entries.size() > kRelayLayoutMaxParams) return;
    RelayParamEntries e;
    e.count = entries.size();
    for (size_t i = 0; i < entries.size(); ++i) {
        e.ordinals[i] = entries[i][0];
        e.offsets[i] = entries[i][1];
        e.sizes[i] = entries[i][2];
    }
    g_relay_layouts[kernel_name] = e;
    XDEBG("meta-extend: registered original layout for %s (%zu params)",
          kernel_name, entries.size());
}

bool query_relay_originals(const char *kernel_name, unsigned int *ordinals,
                           unsigned int *offsets, unsigned int *sizes,
                           size_t max_entries, size_t *out_entries)
{
    std::lock_guard<std::mutex> lock(g_relay_layout_mutex);
    auto it = g_relay_layouts.find(kernel_name);
    if (it == g_relay_layouts.end()) return false;
    const RelayParamEntries &e = it->second;
    if (ordinals != nullptr && offsets != nullptr && sizes != nullptr &&
        max_entries >= e.count) {
        for (size_t i = 0; i < e.count; ++i) {
            ordinals[i] = e.ordinals[i];
            offsets[i] = e.offsets[i];
            sizes[i] = e.sizes[i];
        }
    }
    *out_entries = e.count;
    return true;
}

// Identify the registered ORIGINAL layout of a grown kernel from its
// LOADED (grown) layout, without any name dependency. The meta extend
// keeps the KPARAM record count and every ordinal offset unchanged and
// inflates exactly one ordinal's size so that its blob-relative end
// equals kMetaParamSize (0x1520); all other sizes stay original. A
// registered entry therefore matches when count and offsets agree and
// every registered size is <= the loaded size, with zero diffs (no-op:
// loaded already equals the original; marshaling is unchanged) or with
// exactly one strictly-grown ordinal whose end is exactly the window
// extent. Several different-value one-diff candidates mean the shape is
// ambiguous: refuse rather than guess (value-identical duplicates are
// equivalent and fine).
bool find_relay_originals_by_layout(
    unsigned int count, const unsigned int *offsets, const unsigned int *sizes,
    size_t max_entries, size_t *out_entries,
    unsigned int *ordinals_out, unsigned int *offsets_out,
    unsigned int *sizes_out)
{
    static bool ambiguous_warned = false;
    *out_entries = 0;
    if (count == 0 || count > kRelayLayoutMaxParams) return false;
    if (offsets == nullptr || sizes == nullptr || out_entries == nullptr) return false;

    const RelayParamEntries *unique = nullptr;
    const RelayParamEntries *noop = nullptr;
    bool ambiguous = false;
    {
        std::lock_guard<std::mutex> lock(g_relay_layout_mutex);
        for (const auto &kv : g_relay_layouts) {
            const RelayParamEntries &e = kv.second;
            if (e.count != count) continue;
            bool compatible = true;
            size_t diffs = 0, diff_idx = 0;
            for (size_t i = 0; i < count; ++i) {
                if (e.offsets[i] != offsets[i]) { compatible = false; break; }
                if (sizes[i] < e.sizes[i]) { compatible = false; break; }
                if (sizes[i] > e.sizes[i]) { ++diffs; diff_idx = i; }
            }
            if (!compatible) continue;
            if (diffs == 0) { if (noop == nullptr) noop = &e; continue; }
            if (diffs > 1) continue; // the extender grows at most one ordinal
            uint32_t end = offsets[diff_idx] + sizes[diff_idx];
            if ((size_t)end != (size_t)kMetaParamSize) continue;
            if (unique != nullptr) {
                // value-identical registrations (module reloads) are benign
                bool same = unique->count == e.count;
                for (size_t i = 0; same && i < count; ++i) {
                    same = unique->ordinals[i] == e.ordinals[i] &&
                           unique->offsets[i] == e.offsets[i] &&
                           unique->sizes[i] == e.sizes[i];
                }
                if (same) continue;
                ambiguous = true;
                break;
            }
            unique = &e;
        }
    }
    if (ambiguous) {
        if (!ambiguous_warned) {
            ambiguous_warned = true;
            XWARN("meta-extend: relay layout shape ambiguous (%u params); "
                  "falling back to loaded param info", count);
        }
        return false;
    }
    const RelayParamEntries *sel = (unique != nullptr) ? unique : noop;
    if (sel == nullptr) return false;
    if (ordinals_out != nullptr && offsets_out != nullptr && sizes_out != nullptr &&
        max_entries >= count) {
        for (size_t i = 0; i < count; ++i) {
            ordinals_out[i] = sel->ordinals[i];
            offsets_out[i] = sel->offsets[i];
            sizes_out[i] = sel->sizes[i];
        }
    }
    *out_entries = count;
    return true;
}

} // namespace

namespace {

constexpr uint32_t kFatbinMagic = 0xba55ed50u;
constexpr uint32_t kElfMagic    = 0x464c457fu; // "\x7f" "ELF" little-endian

// CUDA runtime fatbin wrapper (fatbinary_section.h __fatBinC_Wrapper_t):
// {u32 magic 0x466243b1 (FATBINC_MAGIC), u32 version (FATBINC_VERSION 1),
// const unsigned long long *data -> the real fatbin container,
// void *filename_or_fatbins}. Static cudart hands this wrapper to the
// module-load entry and the driver follows wrapper->data (root gdb
// specimen: cuLibraryLoadData called from libcudart_static with the
// wrapper at rdi). The owned block reserves 32 bytes for the 24-byte
// wrapper copy (8 zero pad) so the nested container copy behind it stays
// 16-byte aligned; the wrapper copy's data field is retargeted at it.
constexpr uint32_t kFatbinWrapperMagic = 0x466243b1u; // FATBINC_MAGIC
constexpr uint32_t kFatbinWrapperVer1  = 1u;          // FATBINC_VERSION
constexpr size_t   kFatbinWrapSize     = 24;          // sizeof(__fatBinC_Wrapper_t), lp64
constexpr size_t   kFatbinWrapCopy     = 32;          // incl. alignment pad

constexpr uint8_t kNvInfoFmt4 = 0x04;         // {id, paylen, payload...}
constexpr uint8_t kNvInfoFmt3 = 0x03;         // {id, u16 value}  (HVAL)
constexpr uint8_t kNvInfoFmt2 = 0x02;         // {id, u16 value}  (BVAL)
constexpr uint8_t kNvInfoParamCBank   = 0x0a; // EIATTR_PARAM_CBANK (SVAL)
constexpr uint8_t kNvInfoCBankPrmSize = 0x19; // EIATTR_CBANK_PARAM_SIZE (HVAL)
constexpr uint8_t kNvInfoKParamInfo   = 0x17; // EIATTR_KPARAM_INFO (app args)
constexpr uint8_t kNvInfoKParamInfoV2 = 0x45; // EIATTR_KPARAM_INFO_V2

// EIATTR_KPARAM_INFO (fmt 4, payload 12 bytes) - legacy shape:
//   {u32 index, u16 ordinal, u16 offset, u16 attrs (0xf000 scal /
//   0xf500 ptr), u16 size_field}. Every observed record encodes
//   size_field == size * 4 + 1 (5 records, sizes 4 and 8: 0x11, 0x21;
//   both formulas size*4+1 and (size>>2)<<4|1 coincide for 4-aligned
//   sizes). Offsets are relative to the param base (0x380), the same
//   coordinate system as the launch blob, so the max ordinal end is the
//   declared launch-parameter extent.
// EIATTR_KPARAM_INFO_V2 (fmt 4, payload 12 bytes) - the shape nvcc emits
//   for kernels with a large by-value parameter (kpspec big_kernel,
//   0x1520-byte struct): {u32 index, u16 ordinal, u16 offset, u16
//   plain_size, u16 attrs; attrs 0x0500 pointer / 0x0000 scalar}. The
//   specimen uses V2 records exclusively for this kernel and legacy
//   records for its small sibling, both under CUDA_API_VERSION 0x81.
// kv7jHZ/HX4CV3 evidence: a 0x1520-byte CU_LAUNCH_PARAM_BUFFER is
// rejected with 701 while a 0x20 buffer is accepted on the same patched
// image, so the ordinal table (not the CBANK records alone) governs the
// launch-buffer acceptance candidate; the pass C growth targets it.
constexpr size_t kNvInfoKParamPaylen  = 12;   // both record shapes

// paylen no larger than a .nv.info record can sanity-carry; anything above
// is treated as a parse anomaly rather than trusted (prevents an unwalkable
// record from crashing the section scan).
constexpr uint64_t kNvInfoMaxPaylen = 0x8000;

constexpr uint64_t kRectMagic   = 0x474d495458455847ull; // "XGEXTIMG"
constexpr size_t   kRectHdrSize = 16;                    // magic u64 + size u64

constexpr size_t kImageMaxBytes = (size_t)1 << 27; // sanity bound: 128 MiB

bool meta_extend_enabled()
{
    static const bool enabled = std::getenv("XG_NATIVE_META_EXTEND") != nullptr;
    return enabled;
}

// Second independent knob: also grow the KPARAM_INFO ordinal table so the
// declared launch-parameter extent covers the relay blob (see pass C).
// Separate from XG_NATIVE_META_EXTEND so a metadata-only control run stays
// available. Only meaningful together with blob-form launches; see the
// constraint note in the pass C comments.
bool meta_kparam_enabled()
{
    static const bool enabled = std::getenv("XG_NATIVE_META_KPARAM") != nullptr;
    return enabled;
}

uint64_t le(const char *p, unsigned n)
{
    uint64_t v = 0;
    for (unsigned i = 0; i < n; ++i)
        v |= (uint64_t)(uint8_t)p[i] << (8 * i);
    return v;
}

struct Range
{
    size_t off;
    size_t size;
};

// Range with the kernel's mangled name (from the per-kernel nvinfo section
// name). The name is the key the HAL constructor consults through
// XgGetRelayOriginalParams to marshal the app's arguments with their
// ORIGINAL sizes after the module was loaded with a grown ordinal table.
struct NamedRange : Range
{
    std::string name;
};

// ELF64 little-endian only (CUDA cubins are always ELF64 LE).
bool elf64le(const char *b, size_t size)
{
    return size >= 64 && le(b, 4) == kElfMagic && b[4] == 2 && b[5] == 1;
}

// Whole-file extent a CUDA loader must map: the greater of the ELF header,
// the section table, the program header table, and every section body. nvcc
// cubins place the program header table after the section table (readelf on
// the 11136-byte specimen: section headers end at 10856, program headers at
// 10856..11136), so the section-table end alone truncates real images.
bool cubin_whole_size(const char *b, size_t size, size_t *whole)
{
    if (!elf64le(b, size)) return false;
    uint64_t shoff = le(b + 0x28, 8);
    uint16_t shent = (uint16_t)le(b + 0x3a, 2);
    uint16_t shnum = (uint16_t)le(b + 0x3c, 2);
    uint64_t phoff = le(b + 0x20, 8);
    uint16_t phent = (uint16_t)le(b + 0x36, 2);
    uint16_t phnum = (uint16_t)le(b + 0x38, 2);
    uint64_t end = 64;
    if (shoff != 0 && shnum != 0 && shent >= 0x40)
        end = std::max<uint64_t>(end, shoff + (uint64_t)shnum * shent);
    if (phoff != 0 && phnum != 0 && phent != 0)
        end = std::max<uint64_t>(end, phoff + (uint64_t)phnum * phent);
    // section bodies referenced by in-bounds headers
    if (shoff != 0 && shnum != 0 && shent >= 0x40 &&
        shoff <= size && shnum <= (size - shoff) / shent) {
        for (uint16_t i = 0; i < shnum; ++i) {
            const char *h = b + shoff + (uint64_t)i * shent;
            uint64_t s_off  = le(h + 0x18, 8);
            uint64_t s_size = le(h + 0x20, 8);
            if (s_off <= size && s_size <= size - s_off)
                end = std::max<uint64_t>(end, s_off + s_size);
        }
    }
    *whole = (size_t)end;
    return true;
}

// Whole-container extent of a fatbin. Measured nvcc 12.9 layout: container
// header 0x10 bytes, then kind-2 entries; the fileSize field (0x2be0) counts
// after the container header, entry extent is entryHdrSize + paddedPayload,
// and the whole image is hdrsz + fileSize (0x2bf0). A legacy container whose
// fileSize already includes the header is accepted with a warning.
bool fatbin_whole_size(const char *b, size_t size, size_t *whole,
                       bool *legacy)
{
    uint16_t hdrsz = (uint16_t)le(b + 6, 2);
    uint64_t fsize = le(b + 8, 8);
    if (hdrsz >= 0x10 && fsize >= hdrsz &&
        hdrsz + fsize <= size && hdrsz + fsize >= (uint64_t)hdrsz * 2) {
        *whole = (size_t)(hdrsz + fsize);
        *legacy = false;
        return true;
    }
    if (hdrsz >= 0x10 && fsize >= hdrsz && fsize <= size) {
        XWARN("meta-extend: fatbin fileSize 0x%llx spans past the "
              "header-relative extent (image 0x%llx); using legacy extent",
              (unsigned long long)fsize, (unsigned long long)size);
        *whole = (size_t)fsize;
        *legacy = true;
        return true;
    }
    XWARN("meta-extend: fatbin extents (hdr 0x%x file 0x%llx image 0x%llx) "
          "inconsistent; leaving the image unpatched",
          (unsigned)hdrsz, (unsigned long long)fsize,
          (unsigned long long)size);
    return false;
}

// Collect the per-kernel nvinfo sections (names beginning ".nv.info.", one
// section per kernel) and, optionally, the per-kernel bank DATA sections
// (".nv.constant0." + kernel name) that pass D grows. The module-wide
// ".nv.info" and the ".nv.merc.*" mirrors carry no PARAM_CBANK records and
// are deliberately left out.
bool elf_kernel_info_sections(const char *base, size_t size,
                              std::vector<NamedRange> *nvinfo_out,
                              std::vector<NamedRange> *cs0_out = nullptr)
{
    if (!elf64le(base, size)) return false;
    uint64_t shoff = le(base + 0x28, 8);
    uint16_t shentsize = (uint16_t)le(base + 0x3a, 2);
    uint16_t shnum = (uint16_t)le(base + 0x3c, 2);
    uint16_t shstrndx = (uint16_t)le(base + 0x3e, 2);
    if (shoff == 0 || shnum == 0 || shentsize < 0x40 || shstrndx >= shnum)
        return false;
    if (shoff > size || shnum > (size - shoff) / shentsize)
        return false;
    const char *shstr_hdr = base + shoff + (uint64_t)shstrndx * shentsize;
    uint64_t str_off = le(shstr_hdr + 0x18, 8);
    uint64_t str_size = le(shstr_hdr + 0x20, 8);
    if (str_off > size || str_size > size - str_off)
        return false;
    const char *shstr = base + str_off;
    size_t str_left = (size_t)(size - str_off);
    for (uint16_t i = 0; i < shnum; ++i) {
        const char *h = base + shoff + (uint64_t)i * shentsize;
        uint64_t s_off = le(h + 0x18, 8);
        uint64_t s_size = le(h + 0x20, 8);
        if (s_off > size || s_size > size - s_off) return false;
        uint64_t n_ofs = le(h, 4);
        if (n_ofs >= str_left) continue;
        const char *name = shstr + n_ofs;
        size_t name_left = str_left - (size_t)n_ofs;
        bool is_info = name_left >= 10 &&
                       std::memcmp(name, ".nv.info.", 9) == 0;
        bool is_cs0 = cs0_out != nullptr && name_left >= 15 &&
                      std::memcmp(name, ".nv.constant0.", 14) == 0;
        if (!is_info && !is_cs0) continue;
        char buf[256];
        size_t n = 0;
        while (n + 1 < sizeof(buf) && name[n] != '\0') buf[n] = name[n], ++n;
        buf[n] = '\0';
        // kernel mangled name: the section name minus its prefix
        size_t prefix = is_info ? 9 : 14;
        NamedRange r{(size_t)s_off, (size_t)s_size, std::string(buf + prefix)};
        if (is_info) {
            nvinfo_out->push_back(r);
            if (s_size >= 12) {
                XDEBG("meta-extend: scan %s (0x%llx B)", buf,
                      (unsigned long long)s_size);
            }
        } else {
            cs0_out->push_back(r);
        }
    }
    return true;
}

// Grow one kernel's .nv.constant0 bank DATA section to target bytes by
// zero insertion at the section end, then repack the ELF64 image in
// place: e_phoff/e_shoff and every later section's sh_offset move by the
// 16-byte-rounded shift, the grown section's sh_size grows by the EXACT
// (unrounded) delta, and PT_LOAD program headers covering the insertion
// point gain the shift in p_filesz/p_memsz (later segments move their
// p_offset). The shift is a multiple of 16 and observed followers (the
// largest sh_addralign after constant0 is 16) stay alignment-clean; the
// rounded remainder between the section end and the next section start
// is an inter-section gap, which ELF permits. `span` is the caller's
// writable extent (fatbin entry payload end; owned-copy end for a bare
// cubin) so the tail memmove stays inside the owned buffer. The reserve
// for the shift is the caller's; on structural anomaly nothing is moved
// and false is returned (fail-open: the caller keeps the records-only
// state it already wrote).
bool grow_constant0(char *base, size_t span, size_t c0_off, size_t c0_size,
                    size_t target, size_t *out_shift)
{
    *out_shift = 0;
    if (!elf64le(base, span)) return false;
    if (c0_off > span || c0_size == 0 || c0_size > span ||
        c0_off + c0_size > span || c0_off < 64) {
        XWARN("meta-extend: constant0 extent 0x%zx@0x%zx outside the image",
              c0_size, c0_off);
        return false;
    }
    if (target == c0_size) return true; // already exact
    if (target < c0_size) return true;  // bank image already larger
    size_t delta_exact = target - c0_size;
    size_t shift = align_up16(delta_exact);

    const uint64_t shoff0 = le(base + 0x28, 8);
    const uint16_t shent = (uint16_t)le(base + 0x3a, 2);
    const uint16_t shnum = (uint16_t)le(base + 0x3c, 2);
    const uint64_t phoff0 = le(base + 0x20, 8);
    const uint16_t phent = (uint16_t)le(base + 0x36, 2);
    const uint16_t phnum = (uint16_t)le(base + 0x38, 2);
    if (shoff0 == 0 || shnum == 0 || shent < 0x40 ||
        shoff0 > span || shnum > (span - shoff0) / shent)
        return false;

    // locate the exact constant0 section header entry (match by off+size)
    size_t c0_idx = shnum;
    for (uint16_t i = 0; i < shnum; ++i) {
        const char *h = base + shoff0 + (uint64_t)i * shent;
        if (le(h + 0x18, 8) == c0_off && le(h + 0x20, 8) == c0_size) {
            c0_idx = i;
            break;
        }
    }
    if (c0_idx == shnum) {
        XWARN("meta-extend: no section header entry matches the constant0 "
              "extent 0x%zx@0x%zx", c0_size, c0_off);
        return false;
    }
    { // reject NOBITS (no file bytes to grow) and the odd non-PROGBITS types
        const char *h = base + shoff0 + (uint64_t)c0_idx * shent;
        uint32_t s_type = (uint32_t)le(h + 4, 4);
        if (s_type != 1) {
            XWARN("meta-extend: constant0 section type %u is not PROGBITS; "
                  "left unpatched", s_type);
            return false;
        }
    }

    // move the tail right and zero the inserted hole
    std::memmove(base + c0_off + c0_size + shift,
                 base + c0_off + c0_size, span - (c0_off + c0_size));
    std::memset(base + c0_off + c0_size, 0, shift);

    // ELF header: program/section header tables behind the hole move
    if (phoff0 >= c0_off + c0_size)
        put64(base + 0x20, phoff0 + shift);
    if (shoff0 >= c0_off + c0_size)
        put64(base + 0x28, shoff0 + shift);

    // program headers: every segment strictly behind the hole moves; a
    // PT_LOAD whose file range covers the insertion point grows its size
    const size_t insert = c0_off + c0_size;
    for (uint16_t i = 0; i < phnum; ++i) {
        if (phent < 0x38) break;
        char *p = base + le(base + 0x20, 8) + (uint64_t)i * phent;
        uint32_t p_type = (uint32_t)le(p, 4);
        uint64_t p_off = le(p + 0x08, 8);
        if (p_off >= insert) { put64(p + 0x08, p_off + shift); continue; }
        if (p_type != 1 /* PT_LOAD */) continue;
        uint64_t p_filesz = le(p + 0x20, 8);
        uint64_t p_memsz = le(p + 0x28, 8);
        if (p_filesz == 0 || insert < p_off || insert > p_off + p_filesz)
            continue;
        put64(p + 0x20, p_filesz + shift);
        if (p_memsz >= p_filesz)
            put64(p + 0x28, p_memsz + shift);
        else
            put64(p + 0x28, (uint64_t)(p_filesz + shift));
    }

    // section headers: later sections move; the grown section's size grows
    // by the exact delta (its start is behind the hole, untouched)
    const uint64_t shoff1 = le(base + 0x28, 8);
    for (uint16_t i = 0; i < shnum; ++i) {
        char *h = base + shoff1 + (uint64_t)i * shent;
        uint64_t s_off = le(h + 0x18, 8);
        if (s_off >= insert) {
            put64(h + 0x18, s_off + shift);
        } else if ((size_t)i == c0_idx) {
            put64(h + 0x20, c0_size + delta_exact);
        }
    }
    XINFO("meta-extend: .nv.constant0 0x%zx -> 0x%zx (+0x%zx bytes, tail "
          "shift +0x%zx)", c0_size, target, delta_exact, shift);
    *out_shift = shift;
    return true;
}

// Read-only pre-pass counter shared by both containers: kernels whose
// nvinfo section has an eligible PARAM_CBANK (sm_120 page base, extent
// below the raised one) will take pass B/C and, under META_KPARAM, a
// constant0 pass D with a bounded worst-case shift. Overcounts (walk
// failures later are ignored), never undercounts.
size_t probe_extendable_kernels(const char *base, size_t size)
{
    size_t count = 0;
    auto count_elf = [&](const char *elf, size_t span) {
        if (span < 64 || span > kImageMaxBytes) return;
        if (!elf64le(elf, span)) return;
        uint64_t shoff = le(elf + 0x28, 8);
        uint16_t shent = (uint16_t)le(elf + 0x3a, 2);
        uint16_t shnum = (uint16_t)le(elf + 0x3c, 2);
        uint16_t shstrndx = (uint16_t)le(elf + 0x3e, 2);
        if (shoff == 0 || shnum == 0 || shent < 0x40 || shstrndx >= shnum ||
            shoff > span || shnum > (span - shoff) / shent)
            return;
        const char *shstr_hdr = elf + shoff + (uint64_t)shstrndx * shent;
        uint64_t str_off = le(shstr_hdr + 0x18, 8);
        uint64_t str_size = le(shstr_hdr + 0x20, 8);
        if (str_off > span || str_size > span - str_off) return;
        const char *shstr = elf + str_off;
        size_t str_left = (size_t)(span - str_off);
        for (uint16_t i = 0; i < shnum; ++i) {
            const char *h = elf + shoff + (uint64_t)i * shent;
            uint64_t s_off = le(h + 0x18, 8);
            uint64_t s_size = le(h + 0x20, 8);
            if (s_off > span || s_size > span - s_off) return;
            uint64_t n_ofs = le(h, 4);
            if (n_ofs >= str_left) continue;
            const char *name = shstr + n_ofs;
            size_t name_left = str_left - (size_t)n_ofs;
            if (name_left < 10 ||
                std::memcmp(name, ".nv.info.", 9) != 0)
                continue;
            if (s_size < 4 + 8 + 4) continue; // one PARAM_CBANK record at least
            const char *sec = elf + s_off;
            bool eligible = false;
            size_t o = 0;
            while (o + 4 <= s_size) {
                uint8_t fmt = (uint8_t)sec[o];
                uint8_t id = (uint8_t)sec[o + 1];
                if (fmt != kNvInfoFmt4) {
                    o += 4;
                    continue;
                }
                uint64_t paylen = (uint64_t)le(sec + o + 2, 2);
                if (paylen > kNvInfoMaxPaylen || o + 4 + paylen > s_size)
                    return; // anomaly: stop counting this section
                if (id == kNvInfoParamCBank && paylen == 8) {
                    uint16_t pstart = (uint16_t)le(sec + o + 8, 2);
                    uint16_t psize = (uint16_t)le(sec + o + 10, 2);
                    if (pstart == kMetaParamStart && psize < kMetaParamSize)
                        eligible = true;
                }
                o += (size_t)4 + paylen;
            }
            if (eligible) ++count;
        }
    };
    if (le(base, 4) == kFatbinMagic) {
        size_t whole = 0;
        bool legacy = false;
        if (!fatbin_whole_size(base, size, &whole, &legacy) || whole > size)
            return 0;
        uint16_t hdrsz = (uint16_t)le(base + 6, 2);
        size_t off = hdrsz;
        while (off + 0x20 <= whole) {
            const char *e = base + off;
            uint16_t kind = (uint16_t)le(e, 2);
            uint32_t ehdr = (uint32_t)le(e + 4, 4);
            uint64_t padded = le(e + 8, 8);
            if (ehdr < 0x28 || off + ehdr + padded > whole) break;
            if (kind == 2 &&
                std::memcmp(e + ehdr, "\x7f" "ELF", 4) == 0)
                count_elf(e + ehdr, (size_t)padded);
            off += (size_t)ehdr + (size_t)padded;
        }
    } else {
        count_elf(base, size);
    }
    return count;
}

// Patch the parameter-extent nvinfo records of one kernel section.
//
// Pass A (read-only) finds EIATTR_PARAM_CBANK and decides eligibility from
// its param_start/param_size. Pass B raises EIATTR_CBANK_PARAM_SIZE (the
// record precedes PARAM_CBANK in compiler layout) and PARAM_CBANK itself.
//
// Returns: -1 parse anomaly, 0 no eligible bank, 1 patched, 2 already
// covering the window.
int extend_kernel_info_section(const char *tag, const char *kernel_name,
                               char *base,
                               size_t /*sec_walk_span*/, size_t sec_off,
                               size_t sec_size, bool *wrote,
                               bool *grew_relay)
{
    *grew_relay = false;
    if (sec_size < 4) return 0;
    const char *sec = base + sec_off;

    // pass A: find PARAM_CBANK, read param region
    size_t cbank_rec = sec_size; // offset of the PARAM_CBANK record
    bool cbank_present = false;
    uint16_t pstart = 0, psize = 0;
    size_t o = 0;
    while (o + 4 <= sec_size) {
        uint8_t fmt = (uint8_t)sec[o];
        uint8_t id = (uint8_t)sec[o + 1];
        if (fmt != kNvInfoFmt4 && fmt != kNvInfoFmt3 && fmt != kNvInfoFmt2)
            return -1;
        uint64_t paylen = (fmt == kNvInfoFmt4) ? (uint64_t)le(sec + o + 2, 2) : 0;
        uint64_t rec = (fmt == kNvInfoFmt4) ? 4 + paylen : 4;
        if (rec > kNvInfoMaxPaylen || o + rec > sec_size) return -1;
        if (fmt == kNvInfoFmt4 && id == kNvInfoParamCBank && paylen == 8) {
            cbank_present = true;
            cbank_rec = o;
            pstart = (uint16_t)le(sec + o + 8, 2);
            psize  = (uint16_t)le(sec + o + 10, 2);
        }
        o += (size_t)rec;
    }
    if (o != sec_size) {
        // trailing halves of records: tolerated on a read-only pass only
        // when nothing to patch; otherwise flag the anomaly
        return cbank_present ? -1 : 0;
    }
    if (!cbank_present || pstart != kMetaParamStart) return 0;
    if (psize >= kMetaParamSize) {
        // already extended on a previous load of the same image; make the
        // paired CBANK_PARAM_SIZE record consistent before reporting
        bool aligned = true;
        o = 0;
        while (o + 4 <= sec_size) {
            uint8_t fmt = (uint8_t)sec[o];
            if (fmt == kNvInfoFmt3 &&
                (uint8_t)sec[o + 1] == kNvInfoCBankPrmSize) {
                uint16_t v = (uint16_t)le(sec + o + 2, 2);
                if (v < kMetaParamSize) {
                    if (tag[0] != '\0' && v != 0) {
                        XWARN("meta-extend: %s CBANK_PARAM_SIZE 0x%x "
                              "unextended beside PARAM_CBANK 0x%x",
                              tag, (unsigned)v, (unsigned)psize);
                    }
                    aligned = false;
                    break;
                }
            }
            o += 4;
        }
        return aligned ? 2 : -1;
    }

    // pass B: raise CBANK_PARAM_SIZE (id 0x19, u16 at record offset +2)
    char *buf = base + sec_off;
    o = 0;
    int patched = 0;
    while (o + 4 <= sec_size) {
        uint8_t fmt = buf[o];
        uint8_t id = buf[o + 1];
        if (fmt == kNvInfoFmt4) {
            uint64_t paylen = (uint64_t)le(buf + o + 2, 2);
            if (o + 4 + paylen > sec_size) break; // pass A already validated
            if (id == kNvInfoParamCBank && paylen == 8 && o == cbank_rec) {
                char *field = buf + o + 10;
                field[0] = (char)(kMetaParamSize & 0xff);
                field[1] = (char)((kMetaParamSize >> 8) & 0xff);
                XINFO("meta-extend: %s PARAM_CBANK param_size 0x%x -> 0x%x",
                      tag, (unsigned)psize, (unsigned)kMetaParamSize);
                ++patched;
            }
            o += (size_t)(4 + paylen);
        } else {
            if (id == kNvInfoCBankPrmSize && patched >= 0) {
                uint16_t v = (uint16_t)le(buf + o + 2, 2);
                if (v != kMetaParamSize) {
                    buf[o + 2] = (char)(kMetaParamSize & 0xff);
                    buf[o + 3] = (char)((kMetaParamSize >> 8) & 0xff);
                    XINFO("meta-extend: %s CBANK_PARAM_SIZE 0x%x -> 0x%x",
                          tag, (unsigned)v, (unsigned)kMetaParamSize);
                    ++patched;
                }
            }
            o += 4;
        }
    }
    // pass C (opt-in XG_NATIVE_META_KPARAM): convert the legacy KPARAM_INFO
    // ordinal records into the compiler's own large-parameter shape and grow
    // the max declared ordinal end so it covers the relay blob extent
    // (0x1520 blob-relative). kv7jHZ/HX4CV3 evidence: the driver rejects a
    // 0x1520-byte relay CU_LAUNCH_PARAM_BUFFER with 701 while a 0x20 buffer
    // is accepted on the same patched image (CBANK records raised, original
    // entry), so the KPARAM-declared ordinal extent is the remaining
    // declared-extent candidate governing launch-buffer acceptance.
    // Record conversion follows the nvcc large-parameter specimen: legacy
    // EIATTR_KPARAM_INFO (id 0x17, {index, ord, off, u16 attrs, u16
    // size*4+1}) is rewritten in place (slot length preserved) as
    // EIATTR_KPARAM_INFO_V2 (id 0x45, {index, ord, off, u16 plain_size,
    // u16 attrs}); attrs bits 12..15 are cleared (0xf500 -> 0x0500 pointer,
    // 0xf000 -> 0x0000 scalar) and the grown ordinal carries its plain size.
    // HOST-MARSHALING CONSTRAINT: the constructor deep copy in the HAL
    // marshals per-ordinal from the loaded (grown) param info and would
    // copy the grown size from the app's own argument pointer. The original
    // per-ordinal layout is therefore registered by the extender at patch
    // time (kernels with grown sections are keyed by their mangled kernel
    // name) and the HAL constructor/relay marshaling consults
    // XgGetRelayOriginalParams to allocate and copy ORIGINAL sizes; grown
    // sizes only affect the driver's launch-side blob validation. Non-
    // relay/plain launches never marshal grown sizes with this wiring.
    if (meta_kparam_enabled()) {
        bool walked = true;
        bool have = false;
        size_t best_off = 0, best_rec = 0;
        uint32_t best_end = 0;
        unsigned kordinal = 0;
        std::vector<std::array<uint32_t, 3>> origs;
        o = 0;
        while (o + 4 <= sec_size) {
            uint8_t fmt = (uint8_t)buf[o];
            uint8_t id = (uint8_t)buf[o + 1];
            if (fmt == kNvInfoFmt4) {
                uint64_t paylen = (uint64_t)le(buf + o + 2, 2);
                if (o + 4 + paylen > sec_size) { walked = false; break; }
                if (paylen == kNvInfoKParamPaylen &&
                    (id == kNvInfoKParamInfo ||
                     id == kNvInfoKParamInfoV2)) {
                    // 0x17: {u32 index, u16 ord, u16 off, u16 attrs,
                    //         u16 size*4+1}
                    // 0x45: {u32 index, u16 ord, u16 off, u16 size,
                    //         u16 attrs}
                    bool legacy = (id == kNvInfoKParamInfo);
                    uint16_t ord   = (uint16_t)le(buf + o + 8, 2);
                    uint16_t off16 = (uint16_t)le(buf + o + 10, 2);
                    uint32_t sz, attrs;
                    if (legacy) {
                        uint16_t enc = (uint16_t)le(buf + o + 14, 2);
                        sz    = ((enc & 3) == 1)
                              ? (uint32_t)((enc - 1) >> 2)
                              : 0xffffffffu; // foreign encoding: sized 0
                        attrs = (uint32_t)(le(buf + o + 12, 2) & 0x0fff);
                    } else {
                        sz    = (uint32_t)le(buf + o + 12, 2);
                        attrs = (uint32_t)le(buf + o + 14, 2);
                    }
                    if (sz != 0xffffffffu)
                        origs.push_back({(uint32_t)ord, (uint32_t)off16, sz});
                    if (legacy && sz != 0xffffffffu) {
                        // in-place conversion to the V2 shape: plain size
                        // and cleared attr high nibble
                        char *dst = buf + o + 12;
                        dst[0] = (char)(sz & 0xff);
                        dst[1] = (char)((sz >> 8) & 0xff);
                        dst[2] = (char)(attrs & 0xff);
                        dst[3] = (char)((attrs >> 8) & 0xff);
                        buf[o + 1] = (char)kNvInfoKParamInfoV2;
                        XDEBG("meta-extend: %s KPARAM ordinal %u size 0x%x"
                              " attrs 0x%x: legacy record -> V2", tag,
                              (unsigned)ord, (unsigned)sz, (unsigned)attrs);
                    }
                    if (sz != 0xffffffffu &&
                        (!have ||
                         (uint32_t)off16 + sz > best_end ||
                         ((uint32_t)off16 + sz == best_end &&
                          off16 > best_off))) {
                        have = true;
                        best_end = (uint32_t)off16 + sz;
                        best_off = off16;
                        best_rec = o;
                        kordinal = (unsigned)ord;
                    }
                }
                o += (size_t)4 + paylen;
            } else {
                o += 4;
            }
        }
        if (!walked || o != sec_size) {
            XWARN("meta-extend: %s KPARAM walk anomaly; ordinal table left "
                  "unchanged", tag);
        } else if (!have) {
            XWARN("meta-extend: %s no KPARAM_INFO record found; ordinal "
                  "table left unchanged", tag);
        } else if (best_end >= kMetaParamSize) {
            XDEBG("meta-extend: %s KPARAM extent 0x%x already covers the "
                  "window", tag, (unsigned)best_end);
        } else {
            uint32_t new_size = (uint32_t)(kMetaParamSize - (uint32_t)best_off);
            // plain-size field: 16-bit, so the grown ordinal must fit
            if (new_size > 0xffff) {
                XWARN("meta-extend: %s KPARAM growth 0x%x overflows the "
                      "plain-size field; ordinal table left unchanged", tag,
                      (unsigned)new_size);
            } else {
              buf[best_rec + 12] = (char)(new_size & 0xff);
                buf[best_rec + 13] = (char)((new_size >> 8) & 0xff);
                XINFO("meta-extend: %s KPARAM(V2) ordinal %u offset 0x%x "
                      "size 0x%x -> 0x%x (declared launch extent 0x%x)",
                      tag, kordinal, (unsigned)best_off,
                      (unsigned)(best_end - best_off), (unsigned)new_size,
                      (unsigned)kMetaParamSize);
                ++patched;
                *grew_relay = true;
                // registry entries are ordinal-indexed (the HAL marshals
                // params[ordinal] positionally): CUDA param ordinals are
                // dense 0..n-1; sort and verify before registering
                std::sort(origs.begin(), origs.end(),
                          [](const std::array<uint32_t, 3> &a,
                             const std::array<uint32_t, 3> &b)
                          { return a[0] < b[0]; });
                bool dense = true;
                for (size_t i = 0; i < origs.size(); ++i)
                    if (origs[i][0] != i) { dense = false; break; }
                if (dense) {
                    remember_relay_originals(kernel_name, origs);
                } else {
                    XWARN("meta-extend: %s non-dense KPARAM ordinals; "
                          "original layout not registered", kernel_name);
                }
            }
        }
    }

    if (patched == 0) return -1; // pass A said eligible but nothing was patched
    *wrote = true;
    XINFO("meta-extend: %s declared c[0x0] param region [0x%x, 0x%x]",
          tag, (unsigned)kMetaParamStart,
          (unsigned)(kMetaParamStart + kMetaParamSize));
    return 1;
}

// Patch all eligible kernels of one cubin ELF in place. Returns false on
// structural anomalies (caller then uses the unpatched copy as-is; nothing
// aborts the load).
// Patch all eligible kernels of one cubin ELF in place: the nvinfo extent
// records (pass A/B) and, under META_KPARAM, the KPARAM ordinal table
// (pass C) are rewritten first — the record offsets are all collected
// before any byte moves — then pass D grows the constant0 banks of exactly
// the kernels whose ordinal table grew (descending section order so each
// insertion's tail move does not invalidate the remaining section list;
// the grower re-derives all table positions per call anyway). `movable_span`
// is the byte extent that a constant0 insertion may shift: for a bare
// cubin the owned-copy slack behind `size` already absorbs the shift, for
// a fatbin entry it must reach the CURRENT container end so already-
// processed later entries move coherently (the entry's padded field and
// the container fileSize are then raised by the caller). The shift
// consumption is debited from *reserve_left, which the owner of the
// backing buffer sized per probed kernel before the copy; when the debit
// would go negative the growth for that kernel is skipped (fail-open:
// records-only state, warned). Returns false on structural anomalies;
// *patched counts kernels with rewritten records and *grown the total
// byte shift applied.
bool extend_cubin(const char *what, char *base, size_t size,
                  size_t movable_span, int *patched,
                  size_t *reserve_left, size_t *grown)
{
    std::vector<NamedRange> sections;
    std::vector<NamedRange> cs0;
    if (!elf_kernel_info_sections(base, size, &sections, &cs0)) {
        XDEBG("meta-extend: %s is not a supported ELF64 cubin", what);
        return false;
    }
    // constant0 bank of every kernel whose ordinal table grew (first
    // name match wins)
    std::vector<const NamedRange *> grows;
    char tag[128];
    int idx = 0;
    bool wrote = false;
    for (const NamedRange &r : sections) {
        std::snprintf(tag, sizeof(tag), "%s[%d]", what, idx++);
        bool grew = false;
        int rc = extend_kernel_info_section(tag, r.name.c_str(), base, size,
                                            r.off, r.size, &wrote, &grew);
        if (rc < 0) {
            XWARN("meta-extend: %s nvinfo parse anomaly; kernel left "
                  "unpatched", tag);
            continue;
        }
        if (!grew || cs0.empty()) continue;
        const NamedRange *c0 = nullptr;
        for (const NamedRange &c : cs0) {
            if (c.name == r.name) { c0 = &c; break; }
        }
        if (c0 == nullptr) {
            XWARN("meta-extend: %s ordinal table grew but no .nv.constant0."
                  "%s section exists; bank image left small", tag,
                  r.name.c_str());
        } else {
            grows.push_back(c0);
        }
    }
    *patched = wrote ? 1 : 0;
    if (grows.empty()) return true;

    // descending constant0 offsets: each insertion shifts everything after
    // it, so process the highest bank first; the grower re-derives all
    // table positions per call. The movable span grows by every shift:
    // earlier growths already extended the image tail, and a later
    // insertion must move that WHOLE extended tail (tables included), so
    // the span is re-based after each growth, not fixed at the original.
    std::sort(grows.begin(), grows.end(),
              [](const NamedRange *a, const NamedRange *b)
              { return a->off > b->off; });
    size_t grow_span = movable_span;
    for (const NamedRange *g : grows) {
        size_t shift = 0;
        if (*reserve_left < kConst0GrowReserve) {
            XWARN("meta-extend: owned-copy reserve exhausted before %s "
                  ".nv.constant0 %s; bank image left small", what,
                  g->name.c_str());
            continue;
        }
        if (!grow_constant0(base, grow_span, g->off, g->size,
                            kMetaConst0Target, &shift)) {
            XWARN("meta-extend: %s .nv.constant0 %s growth failed; bank "
                  "image left small", what, g->name.c_str());
            continue;
        }
        *reserve_left -= shift;
        *grown += shift;
        grow_span += shift;
    }
    return true;
}

// Copy-and-patch the whole image held in data[0..size). The owned buffer
// extends beyond `size` by the probe-sized slack (*reserve_left on entry,
// debited as constant0 growth consumes it). Returns the total byte shift
// applied through *grown.
void meta_extend_image(char *data, size_t size, size_t *reserve_left,
                       size_t *grown)
{
    if (le(data, 4) != kFatbinMagic) {
        int kernels = 0;
        extend_cubin("cubin", data, size, size, &kernels, reserve_left,
                     grown);
        if (kernels == 0) {
            XDEBG("meta-extend: cubin untouched (no eligible kernel)");
        }
        return;
    }

    size_t whole = 0;
    bool legacy = false;
    // header fields were validated during the whole-size computation in
    // XgMetaExtendImage; re-derive the walk bound from the copy itself
    if (!fatbin_whole_size(data, size, &whole, &legacy) || whole > size ||
        whole >= kImageMaxBytes) {
        XWARN("meta-extend: fatbin bound re-check failed; loading unpatched");
        return;
    }
    // collect every kind-2 ELF entry first, then patch the payloads from
    // the LAST entry back to the first: a constant0 growth moves the
    // container tail (all later entries) coherently, which is only safe
    // when those entries are already processed
    struct FatEntry { size_t off; uint32_t hdr; uint64_t padded; };
    std::vector<FatEntry> entries;
    {
        uint16_t hdrsz = (uint16_t)le(data + 6, 2);
        size_t off = hdrsz;
        while (off + 0x20 <= whole) {
            const char *e = data + off;
            uint16_t kind = (uint16_t)le(e, 2);
            uint32_t ehdr = (uint32_t)le(e + 4, 4);
            uint64_t padded = le(e + 8, 8);
            if (ehdr < 0x28 || off + ehdr + padded > whole) {
                XWARN("meta-extend: fatbin entry beyond container (off "
                      "0x%llx); leaving the rest unpatched",
                      (unsigned long long)off);
                break;
            }
            if (kind == 2 &&
                std::memcmp(e + ehdr, "\x7f" "ELF", 4) == 0)
                entries.push_back({off, ehdr, padded});
            off += (size_t)ehdr + (size_t)padded;
        }
    }
    std::sort(entries.begin(), entries.end(),
              [](const FatEntry &a, const FatEntry &b)
              { return a.off > b.off; });
    int kernels = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        const FatEntry &en = entries[i];
        char *elf = data + en.off + en.hdr;
        size_t grown_before = *grown;
        // container current end: grows with every growth so far
        size_t cont_end = (size_t)le(data + 8, 8) +
                          (size_t)le(data + 6, 2);
        int k = 0;
        // per-kernel tags inside one cubin keep the prior "what[kernel]"
        // log shape; entry identity is not needed (nvcc containers hold
        // one sm cubin entry per architecture)
        extend_cubin("fatbin-cubin", elf, (size_t)en.padded,
                     cont_end - (en.off + en.hdr),
                     &k, reserve_left, grown);
        kernels += k;
        size_t delta = *grown - grown_before;
        if (delta != 0) {
            // the payload's file extent grew; raise the entry's
            // padded-payload size and the container fileSize (the grower
            // already moved the trailing container bytes)
            put64(data + en.off + 8, en.padded + delta);
            put64(data + 8, (uint64_t)(cont_end - (size_t)le(data + 6, 2)) +
                                delta);
        }
    }
    if (kernels == 0) {
        XDEBG("meta-extend: fatbin had no eligible kernel");
    }
}

} // namespace

XgExtendedImage XgMetaExtendImage(const void *image)
{
    XgExtendedImage out{image, false};
    if (image == nullptr) return out;
    if (!meta_extend_enabled()) return out;

    const char *b = (const char *)image;
    size_t size = 0;
    bool wrapped = false; // b is the runtime fatbin wrapper; payload at nest
    const char *nest = nullptr;
    if (le(b, 4) == kFatbinMagic) {
        size_t whole = 0;
        bool legacy = false;
        if (!fatbin_whole_size(b, kImageMaxBytes, &whole, &legacy)) {
            // the first probe runs unbounded; verify against a real read
            // only after the caller's buffer size is unknown: use the
            // second stage below
        }
        (void)whole;
        // Container extent must be derived from the copy below; a bare
        // header probe cannot bound the caller's allocation, so size the
        // buffer from the fatbin header itself and let the staging path
        // validate.
        uint16_t hdrsz = (uint16_t)le(b + 6, 2);
        uint64_t fsize = le(b + 8, 8);
        if (hdrsz < 0x10 || fsize < hdrsz ||
            fsize + hdrsz > kImageMaxBytes) {
            XWARN("meta-extend: unusable fatbin header (hdr 0x%x size "
                  "0x%llx); loading unpatched", (unsigned)hdrsz,
                  (unsigned long long)fsize);
            return out;
        }
        size = (size_t)(fsize + hdrsz);
    } else if (le(b, 4) == kElfMagic) {
        if (!cubin_whole_size(b, kImageMaxBytes, &size)) {
            XWARN("meta-extend: unusable cubin header; loading unpatched");
            return out;
        }
    } else if (le(b, 4) == kFatbinWrapperMagic &&
               le(b + 4, 4) == kFatbinWrapperVer1) {
        // the runtime fatbin wrapper: own a retargeted wrapper copy plus the
        // patched nested container, in one block (see kFatbinWrapCopy)
        uint64_t dptr = le(b + 8, 8);
        if (dptr == 0) {
            XWARN("meta-extend: fatbin wrapper without data; loading "
                  "unpatched");
            return out;
        }
        const char *d = (const char *)(uintptr_t)dptr;
        if (le(d, 4) == kFatbinMagic) {
            uint16_t hdrsz = (uint16_t)le(d + 6, 2);
            uint64_t fsize = le(d + 8, 8);
            if (hdrsz < 0x10 || fsize < hdrsz ||
                fsize + hdrsz > kImageMaxBytes) {
                XWARN("meta-extend: unusable wrapped fatbin header (hdr "
                      "0x%x size 0x%llx); loading unpatched", (unsigned)hdrsz,
                      (unsigned long long)fsize);
                return out;
            }
            size = (size_t)(fsize + hdrsz);
        } else if (le(d, 4) == kElfMagic) {
            if (!cubin_whole_size(d, kImageMaxBytes, &size)) {
                XWARN("meta-extend: unusable wrapped cubin header; loading "
                      "unpatched");
                return out;
            }
        } else {
            XDEBG("meta-extend: wrapper payload is neither fatbin nor "
                  "cubin; unchanged");
            return out;
        }
        wrapped = true;
        nest = d;
    } else {
        return out; // PTX text or other unknown container: unchanged
    }
    if (size < 64 || size > kImageMaxBytes) {
        XWARN("meta-extend: image extent 0x%llx rejected; loading unpatched",
              (unsigned long long)size);
        return out;
    }

    size_t wrap = wrapped ? kFatbinWrapCopy : 0;
    // pass D reserve: worst-case constant0 shift per pass-C-growable
    // kernel (probe counts eligible nvinfo sections over the SOURCE
    // layout; collection precedes all byte moves in the patcher)
    size_t grown_total = 0;
    size_t reserve = 0;
    if (meta_kparam_enabled()) {
        size_t kernels = probe_extendable_kernels(wrapped ? nest : b, size);
        reserve = kernels * kConst0GrowReserve;
        if (kernels != 0) {
            XINFO("meta-extend: %zu growable kernel(s) probed, constant0 "
                  "reserve 0x%llx bytes", kernels,
                  (unsigned long long)reserve);
        }
    }
    try {
        void *base = nullptr;
        // header-checked allocation: [kRectMagic | copy size | image bytes
        // (+ constant0 growth slack)]; the wrapper case prepends a
        // 32-byte retargeted wrapper copy (24 bytes per
        // fatbinary_section.h plus 8 zero pad) in front of the nested
        // container copy, so one free covers both
        void *block = std::malloc(kRectHdrSize + wrap + size + reserve);
        base = block;
        if (base == nullptr) {
            XWARN("meta-extend: image copy allocation failed; loading "
                  "unpatched");
            return out;
        }
        *(uint64_t *)base = kRectMagic;
        char *data = (char *)base + kRectHdrSize;
        if (wrapped) {
            std::memcpy(data, b, kFatbinWrapSize);
            std::memset(data + kFatbinWrapSize, 0, wrap - kFatbinWrapSize);
            // retarget the copy's data field at the nested fatbin copy
            *(char **)(data + 8) = data + wrap;
            std::memcpy(data + wrap, nest, size);
            meta_extend_image(data + wrap, size, &reserve, &grown_total);
            *((uint64_t *)base + 1) = (uint64_t)(wrap + size + grown_total);
            XINFO("meta-extend: owned fatbin-wrapper copy 0x%llx + 0x%llx "
                  "bytes (constant0 grown +0x%llx, source wrapper 0x%llx)",
                  (unsigned long long)wrap, (unsigned long long)size,
                  (unsigned long long)grown_total,
                  (unsigned long long)(uintptr_t)image);
        } else {
            std::memcpy(data, b, size);
            meta_extend_image(data, size, &reserve, &grown_total);
            *((uint64_t *)base + 1) = (uint64_t)(size + grown_total);
            XINFO("meta-extend: owned copy 0x%llx bytes (constant0 grown "
                  "+0x%llx, source image 0x%llx)",
                  (unsigned long long)size,
                  (unsigned long long)grown_total,
                  (unsigned long long)(uintptr_t)image);
        }
        out.ptr = data;
        out.owned = true;
    } catch (...) {
        XWARN("meta-extend: image copy failed; loading unpatched");
        if (out.owned)
            XgFreeExtendedImage(const_cast<void *>(out.ptr));
        return out;
    }
    return out;
}

void XgFreeExtendedImage(void *p)
{
    if (p == nullptr) return;
    char *hdr = (char *)p - kRectHdrSize;
    uint64_t magic = le(hdr, 8);
    if (magic != kRectMagic) {
        // not an owned copy: never free pointers we did not hand out
        XWARN("meta-extend: pointer 0x%llx is not an owned image copy; "
              "NOT freed", (unsigned long long)(uintptr_t)p);
        return;
    }
    std::free(hdr);
}

} // namespace xsched::cuda

int XgGetRelayOriginalParams(const char *kernel_name, unsigned int *ordinals,
                             unsigned int *offsets, unsigned int *sizes,
                             size_t max_entries, size_t *out_entries)
{
    if (out_entries != nullptr) *out_entries = 0;
    if (kernel_name == nullptr || out_entries == nullptr) return 0;
    if (xsched::cuda::query_relay_originals(kernel_name, ordinals, offsets,
                                            sizes, max_entries, out_entries))
        return 1;
    XDEBG("meta-extend: no registered original layout for %s", kernel_name);
    return 0;
}

int XgHasRelayLayouts(void)
{
    std::lock_guard<std::mutex> lock(xsched::cuda::g_relay_layout_mutex);
    return xsched::cuda::g_relay_layouts.empty() ? 0 : 1;
}

int XgFindRelayOriginalParams(unsigned int count, const unsigned int *offsets,
                              const unsigned int *sizes, unsigned int *ordinals,
                              unsigned int *out_offsets, unsigned int *out_sizes,
                              size_t max_entries, size_t *out_entries)
{
    if (out_entries != nullptr) *out_entries = 0;
    if (out_entries == nullptr) return 0;
    if (xsched::cuda::find_relay_originals_by_layout(
            count, offsets, sizes, max_entries, out_entries,
            ordinals, out_offsets, out_sizes))
        return 1;
    XDEBG("meta-extend: no registered original layout matches loaded shape "
          "(%u params)", count);
    return 0;
}
